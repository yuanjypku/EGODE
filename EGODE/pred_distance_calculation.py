import scipy 
import torch
import numpy as np

"""
object collision checks
    calculate the distance between the yellow and red objects
    and compares with the pre-determined threshold
"""
def pred_distance(positions, instance_idx, red_id, yellow_id, scenario):
    spacing = 0.05
    st, ed = instance_idx[red_id], instance_idx[red_id + 1] # get red and red+1 instance id
    red_pts = positions[st:ed]
    st2, ed2 = instance_idx[yellow_id], instance_idx[yellow_id + 1] # get yellow and yellow+1 instance id
    yellow_pts = positions[st2:ed2]
    sim_mat = scipy.spatial.distance_matrix(yellow_pts, red_pts, p=2)
    min_dist= np.min(sim_mat)

    if "Drape" in scenario:
        thres = 0.1222556027835 * 0.05/0.035
    elif "Contain" in scenario:
        thres = spacing * 1.0
    elif "Drop" in scenario:
        thres = spacing * 1.0
    else:
        thres = spacing * 1.5
    return min_dist < thres

def pred_distance_torch(positions, instance_idx, red_id, yellow_id, scenario):
    spacing = 0.05
    st, ed = instance_idx[red_id], instance_idx[red_id + 1] # get red and red+1 instance id
    red_pts = positions[st:ed]
    st2, ed2 = instance_idx[yellow_id], instance_idx[yellow_id + 1] # get yellow and yellow+1 instance id
    yellow_pts = positions[st2:ed2]

    # sim_mat = scipy.spatial.distance_matrix(yellow_pts, red_pts, p=2)
    diff = yellow_pts.unsqueeze(1) - red_pts.unsqueeze(0)
    sim_mat = torch.norm(diff, dim=-1)

    min_dist= sim_mat.min()

    if "Drape" in scenario:
        thres = 0.1222556027835 * 0.05/0.035
    elif "Contain" in scenario:
        thres = spacing * 1.0
    elif "Drop" in scenario:
        thres = spacing * 1.0
    else:
        thres = spacing * 1.5
    return min_dist < thres
    