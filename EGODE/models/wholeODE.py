import os
import torch
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint_event
from torchdiffeq import odeint_adjoint as odeint
from data_new import recalculate_velocities, denormalize_torch, load_data_dominoes, normalize_torch
from utils_geom import calc_rigid_transform_torch
from torch_scatter import scatter
from pred_distance_calculation import pred_distance_torch

class CoupledODEfunc(nn.Module):
    def __init__(self, coupledGNN, ransac_on_pred, position_dim, debug):
        super().__init__()
        self.coupledGNN = coupledGNN
        self.ransac_on_pred = ransac_on_pred
        self.position_dim = position_dim
        self.debug = debug
    
    def set_forward_params(self, h_p, obj_id, obj_type, stat, dt, phases_dict, instance_idx):
        self.h_p = h_p
        self.obj_id = obj_id
        self.obj_type = obj_type
        self.stat = stat
        self.dt = dt
        self.phases_dict = phases_dict
        self.instance_idx = instance_idx

    def ransac(self, f_p, df_p):
        v = denormalize_torch([df_p[..., :3]], [torch.Tensor(self.stat[1])])[0]
        positions_prev = f_p[..., :3]
        predicted_positions = positions_prev + v * self.dt
        for obj_idx in range(len(self.instance_idx) - 1):
            st, ed = self.instance_idx[obj_idx], self.instance_idx[obj_idx + 1]
            if self.phases_dict['material'][obj_idx] == 'rigid':
                pos_prev = positions_prev[st:ed]
                pos_pred = predicted_positions[st:ed]
                R, T = calc_rigid_transform_torch(pos_prev, pos_pred)
                refined_pos = ((R @ pos_prev.T) + T).T
                predicted_positions[st:ed, :] = refined_pos
        v_new = (predicted_positions - positions_prev)/self.dt 
        df_p_new = torch.cat([v_new, df_p[..., 3:]], dim=-1)
        return df_p_new

    def forward(self, t, z,):
        '''z: torch.cat([f_p, f_o])
        '''

        h_p = self.h_p
        f_p = z[:len(h_p)]
        f_o = z[len(h_p):]

        df_p, df_o = self.coupledGNN(f_p, h_p, f_o, self.obj_id, self.obj_type)
        
        ### use ransac for refining predicted position
        if self.ransac_on_pred:
            df_p_new = self.ransac(f_p, df_p)
            return torch.cat([df_p_new, df_o], dim=-1)
        else:
            return torch.cat([df_p, df_o], dim=-1)

class EventFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 200),
            nn.SiLU(),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x = self.mlp(z)
        return x.mean()

class CoupledWholeODE(nn.Module):
    def __init__(self, 
                 odefunc,
                 training_fpt):
                 
        super().__init__()
        self.odefunc = odefunc
        self.event_func = EventFunc()
        self.training_fpt = training_fpt
    
    def forward(self, total_nframes, start_id, data, p_pred, phases_dict, stat, dt, red_id, yellow_id, scenario, trial_name, timesteps, criterionMSE, data_names, rmse, n_particles):
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.
            
        data_nxt_list = []
        # INITIALIZE
        x0 = torch.Tensor(data[0]).cuda()
        v0 = torch.Tensor(data[1]).cuda()
        f_p0 = torch.cat([x0, v0], dim=0)
        h = torch.zeros((x0.shape[0], 4)).cuda()
        obj_id = torch.zeros_like(x0, dtype=torch.int64)[..., 0]
        obj_type = []
        f_o0 = scatter(f_p0, obj_id, dim=0, reduce='mean')  # [N_obj, 3, 2]
        z0 = torch.cat([f_p0, f_o0], dim=0).reshape(-1, 6)
        instance_idx = phases_dict['instance_idx']
        for i in range(len(instance_idx) - 1):
            obj_id[instance_idx[i]:instance_idx[i + 1]] = i
            obj_type.append(phases_dict['material'][i])
            if phases_dict['material'][i] == 'rigid':
                h[instance_idx[i]:instance_idx[i + 1], 0] = 1
            elif phases_dict['material'][i] in ['cloth', 'fluid']:
                h[instance_idx[i]:instance_idx[i + 1], 1] = 1

        options = {'step_size': dt}
        self.odefunc.set_forward_params(h, obj_id, obj_type, stat, dt, phases_dict, instance_idx,)
        t = torch.arange(total_nframes - start_id + 1, dtype=torch.float32, device=z0.device) * dt  
        
        # ODE
        n_events = 0
        max_events = 20
        event_times = []
        t0 = torch.tensor([0.0]).to(t)
        trajectory = [z0]
        while t0 < t[-1] and n_events < max_events:
            last = n_events == max_events - 1
            if not last:
                event_t, solution = odeint_event(self.odefunc, z0, t, event_fn=self.event_func, odeint_interface=odeint, method='euler', options=options)
            else:
                event_t = t[-1]
            
            interval_ts = t[t > t0]
            interval_ts = interval_ts[interval_ts <= event_t]
            interval_ts = torch.cat([t0.reshape(-1), interval_ts.reshape(-1)])
            solution_ = odeint(
                self.odefunc, z0, interval_ts, atol=1e-8, rtol=1e-8
            )
            traj_ = solution_[0][1:]
            trajectory.append(traj_)

            if event_t < t[-1]:
                z0 = solution[-1]
            
            event_times.append(event_t)
            t0 = event_t

            n_events += 1

        trajectory = torch.cat(trajectory, dim=0).reshape(-1)
        ode_outputs = trajectory

        # EVALUATE
        for current_fid in range(total_nframes - start_id):
            z = ode_outputs[current_fid]
            z_next = ode_outputs[current_fid + 1]
            try:
                data_nxt_path = os.path.join(trial_name, str(timesteps[current_fid + start_id] + int(self.training_fpt)) + '.h5')
                data_nxt = load_data_dominoes(data_names, data_nxt_path, phases_dict)
                data_nxt = [d if d is None else torch.Tensor(d).cuda() for d in data_nxt]
                data_nxt_list.append(data_nxt)
                _, data_nxt = recalculate_velocities([[z[0], z[1]], data_nxt], dt, data_names)
                data_nxt = normalize_torch(data_nxt, [torch.Tensor(i_stat) for i_stat in stat])
                label = data_nxt[1][:n_particles]
                loss = np.sqrt(criterionMSE(z_next[1], label).item())
                rmse.append(loss)
            except:
                pass
        
        for current_fid in range(total_nframes - start_id):
            z = ode_outputs[current_fid]
            p_pred[start_id + current_fid] = z[0].detach().cpu().numpy()
            
            # object collision checks
            z_next = ode_outputs[current_fid + 1]
            positions = z_next[0]
            pred_target_contacting_zone = pred_distance_torch(positions, instance_idx, red_id, yellow_id, scenario)
            if pred_target_contacting_zone:
                pred_is_positive_trial = True
            
        return p_pred, rmse, pred_is_positive_trial
