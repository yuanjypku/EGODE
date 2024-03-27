import os
import sys
import random
import logging
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import CoupledGNN
from models.wholeODE import CoupledODEfunc, CoupledWholeODE

from data_new import collate_fn, PhysicsFleXDatasetODE
from utils import get_query_dir
import time
from datetime import datetime

def get_logger(save_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(save_dir + "/log_100_iter.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


data_root = get_query_dir("dpi_data_dir")
out_root = get_query_dir("out_dir")

parser = argparse.ArgumentParser()
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--time_step_clip', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./60.)
parser.add_argument('--training_fpt', type=float, default=3)

parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--p_step', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=200)

parser.add_argument('--model_name')
parser.add_argument('--floor_cheat', type=int, default=1)
parser.add_argument('--env', default='TDWdominoes')
parser.add_argument('--train_valid_ratio', type=float, default=0.9)
parser.add_argument('--outf', default='Support')
parser.add_argument('--dataf', default='Support,')
parser.add_argument('--statf', default="")
parser.add_argument('--noise_std', type=float, default='0')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--gen_stat', type=int, default=0)

parser.add_argument('--subsample_particles', type=int, default=1)

parser.add_argument('--log_per_iter', type=int, default=10)
parser.add_argument('--ckp_per_iter', type=int, default=10000)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--augment_worldcoord', type=int, default=0)

parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)

parser.add_argument('--n_instance', type=int, default=0)
parser.add_argument('--n_stages', type=int, default=0)
parser.add_argument('--n_his', type=int, default=0)

parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--forward_times', type=int, default=2)

parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--resume_iter', type=int, default=0)

# shape state:
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)
parser.add_argument('--position_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

args = parser.parse_args()

phases_dict = dict()

random.seed(args.seed)
torch.manual_seed(args.seed)
print('Fix seed', args.seed)
print('Training dataset ', args.dataf)
print('Training batchsize ', args.batch_size)
dataset_name = args.dataf.split(',')[0]
model_load_prefix = os.path.join(out_root, 'dump/')

model_saved_name_list = []
# torch.autograd.set_detect_anomaly(True)

# preparing phases_dict
if args.env == "TDWdominoes":
    args.n_rollout = 2# how many data
    data_names = ['positions', 'velocities']
    args.time_step = 200
    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3
    args.dt = 0.01

    # object attr:
    # [rigid, fluid, root_0]
    args.attr_dim = 3

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = -1
    args.time_step = 301
    args.time_step_clip = 0
    args.n_stages = 4
    args.n_stages_types = ["leaf-leaf", "leaf-root", "root-root", "root-leaf"]

    args.neighbor_radius = 0.08

    phases_dict = dict()  # load from data
    # ["root_num"] = [[]]
    # phases_dict["instance"] = ["fluid"]
    # phases_dict["material"] = ["fluid"]
    if ',' in args.dataf and 'ODE' in args.model_name:
        args.outf = args.dataf.split(',')[0] + "_" + args.model_name + "fpt" + str(args.training_fpt)
    else:
        args.outf = args.dataf.split(',')[0] + "_" + args.model_name 
    args.outf = args.outf.strip()
    args.outf = os.path.join(out_root, 'dump/' , args.outf , datetime.now().strftime("%y%m%d%H%M%S"))
else:
    raise AssertionError("Unsupported env")

scenario = args.dataf
label_source_root = get_query_dir("dpi_data_dir")
mode = args.mode
label_file = os.path.join(label_source_root, mode, "labels", f"{scenario}.txt")
gt_labels = []
with open(label_file, "r") as f:
    for line in f:
        trial_name, label = line.strip().split(",")
        gt_labels.append((trial_name[:-5], (label == "True")))
dt = args.training_fpt * args.dt

data_root = os.path.join(data_root, "train")
args.data_root = data_root
if "," in args.dataf:
    # list of folder
    args.dataf = [os.path.join(data_root, tmp.strip()) for tmp in args.dataf.split(",") if tmp != ""]

else:
    args.dataf = args.dataf.strip()
    if "/" in args.dataf:
        args.dataf = 'data/' + args.dataf
    else:  # only prefix
        args.dataf = 'data/' + args.dataf + '_' + args.env
    os.system('mkdir -p ' + args.dataf)
os.system('mkdir -p ' + args.outf)


logger = get_logger(args.outf)
logger.info('Fix seed ' + str(args.seed))

datasets = {phase: PhysicsFleXDatasetODE(
    args, phase, phases_dict, args.verbose_data) for phase in ['train', 'valid']}


for phase in ['train', 'valid']:
    datasets[phase].load_data(args.env)

use_gpu = torch.cuda.is_available()
assert(use_gpu)
device = torch.device("cuda:0" if use_gpu else "cpu")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
assert args.batch_size==1 # batch_size > 1 is not supported yet
if args.batch_size == 1:
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=args.batch_size,
        shuffle=True if x == 'train' else False,
        # num_workers=args.num_workers,
        collate_fn=collate_fn)
        for x in ['train', 'valid']}

# define propagation network

args.noise_std = 3e-4
model = CoupledGNN(n_layer=args.n_layer, s_dim=4, hidden_dim=args.hidden_dim, activation=nn.SiLU(),
                cutoff=0.08, gravity_axis=1, p_step=args.p_step)
odefunc = CoupledODEfunc(
    coupledGNN = model,
    ransac_on_pred = args.ransac_on_pred,
    position_dim = args.position_dim ,
    debug = args.debug,
)
whole_ode = CoupledWholeODE(
    odefunc = odefunc,
    training_fpt = args.training_fpt ,
)



with open(os.path.join(args.outf, "args_stat.pkl"), 'wb') as f:
    import pickle
    pickle.dump(args, f)


optimizer = optim.Adam(whole_ode.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

criterionMSE = nn.MSELoss()

optimizer.zero_grad()
whole_ode = whole_ode.to(device)

st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
best_valid_loss = np.inf
train_iter = 0
current_loss = 0
best_valid_epoch = -1
predicted_nframes = None 

for epoch in range(st_epoch, args.n_epoch):

    phases = ['train', 'valid'] if args.eval == 0 else ['valid']
    for phase in phases:

        whole_ode.train(phase == 'train')
        previous_run_time = time.time()
        start_time = time.time()

        losses = 0.
        for i, data in enumerate(dataloaders[phase]):
                
            # start_time = time.time()
            # print("previous run time", start_time - previous_run_time)
            if (len(data) > 6) :
                x, v, h, obj_id, obj_type, v_target, predicted_nframes = data
            else :
                x, v, h, obj_id, obj_type, v_target = data

            x = torch.Tensor(x).to(device)
            v = torch.Tensor(v).to(device)
            h = torch.Tensor(h).to(device)
            obj_id = torch.LongTensor(obj_id).to(device)
            v_target = torch.Tensor(v_target).to(device)

            total_nframes = x.size()
            start_id = 15 
            stat = [np.zeros((3,3)), np.zeros((3,3))]


            if scenario in ["Dominoes", "Collide", "Drop"]:
                red_id = 1
                yellow_id = 0
            elif scenario in ["Drape"]:
                instance_idx = phases_dict["instance_idx"]
                yellow_id = 0
                red_id = len(instance_idx) - 1 -1
            elif scenario in ["Roll"]:
                yellow_id = 0
                if "ramp" in trial_name:
                    red_id = 2
                else:
                    red_id = 1
            else:
                if "red_id" not in phases_dict:
                    raise RuntimeError()
                red_id = phases_dict["red_id"]
                yellow_id = phases_dict["yellow_id"]
            count_nodes = x.shape[0]
            n_particles = count_nodes
            timesteps  = [t for t in range(0, args.time_step - int(args.training_fpt), int(args.training_fpt))]

            
            rmse = []
            with torch.set_grad_enabled(phase == 'train'):
                p_pred, rmse, pred_is_positive_trial = whole_ode.forward(total_nframes, start_id, data, p_pred, phases_dict, stat, dt, red_id, yellow_id, scenario, trial_name, timesteps, criterionMSE, data_names, rmse, n_particles)

            label = v_target
            loss = torch.mean(rmse)
            current_loss = np.sqrt(loss.item())
            losses += np.sqrt(loss.item())
            if phase == 'train':
                train_iter += 1
                loss.backward()
                if i % args.forward_times == 0 and i!=0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
        losses /= len(dataloaders[phase])

        if phase == 'valid':
            scheduler.step(losses)
            if losses < best_valid_loss:
                best_valid_loss = losses
                best_valid_epoch = epoch

            if epoch - best_valid_epoch >= 10:
                print('Early stopping with 10 epochs!')
                logger.info('Early stopping with 10 epochs!')
                print('Best valid loss {:.6f}, Best valid epoch {:4d}'.format(best_valid_loss, best_valid_epoch))
                logger.info('Best valid loss {:.6f}, Best valid epoch {:4d}'.format(best_valid_loss, best_valid_epoch))
                exit(0)

