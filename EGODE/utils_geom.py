import numpy as np
import torch
import os
import trimesh
import copy
import matplotlib.pyplot as plt


def calc_rigid_transform(XX, YY):
    X = XX.copy().T
    Y = YY.copy().T

    mean_X = np.mean(X, 1, keepdims=True)
    mean_Y = np.mean(Y, 1, keepdims=True)
    X = X - mean_X
    Y = Y - mean_Y
    C = np.dot(X, Y.T)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
    R = np.dot(Vt.T, np.dot(D, U.T))
    T = mean_Y - np.dot(R, mean_X)

    '''
    YY_fitted = (np.dot(R, XX.T) + T).T
    print("MSE fit", np.mean(np.square(YY_fitted - YY)))
    '''

    return R, T


def calc_rigid_transform_torch(XX, YY):
    '''XX, YY: torch.Tensor, shape=(N, 3)
    '''
    X = XX.T
    Y = YY.T

    mean_X = torch.mean(X, 1, keepdim=True)
    mean_Y = torch.mean(Y, 1, keepdim=True)
    X = X - mean_X
    Y = Y - mean_Y

    C = X @ Y.T
    U, _, V = torch.svd(C)

    det_val = torch.det(V @ U.T)
    D = torch.diag(torch.tensor([1, 1, det_val], device=XX.device))
    R = V @ D @ U.T
    T = mean_Y - R @ mean_X

    return R, T

