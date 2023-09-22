import cv2
import numpy as np
import filterpy.common
import filterpy.kalman
import numpy as np
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag


def create_kalman_filter():
    dt = 1/30
    filter = filterpy.kalman.KalmanFilter(dim_x=8, dim_z=6)
    filter.F = np.array(
        [
            [1, dt, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0], 
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ]
    )
    q = Q_discrete_white_noise(dim=2, dt=dt, var=1)
    q1 = 0.000001
    filter.Q = block_diag(q, q, q1, q1, q1, q1)
    filter.H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    filter.R = np.array(
        [
            [50, 0, 0, 0, 0, 0],
            [0, 50, 0, 0, 0, 0],
            [0, 0, 8e02, 0, 0, 0],
            [0, 0, 0, 6.87293869e-02, 0, 0],
            [0, 0, 0, 0, 5.67103028e-02, 0],
            [0, 0, 0, 0, 0, 9.84601130e02]
        ]
    )
    filter.P = block_diag(100, 100, 100, 100, 8, 8, 8, 8)
    return filter


def get_tris_around_tri(tri, idx, level):
    nbrs = [idx]
    for ll in range(level):
        new_nbrs = []
        for nid in nbrs:
            if nid!=-1:
                new_nbrs += tri.neighbors[nid].tolist()
        nbrs += new_nbrs
    nbrs = np.asarray(nbrs)
    nbrs = nbrs[nbrs!=-1]
    return np.unique(nbrs)

def get_point_ids_around_tri(T, idx, level=3, include_src_tri=False):
    if include_src_tri:
        src_ids = np.zeros(3)
    else:
        src_ids = T['simps'][idx].ravel() 
    tris = get_tris_around_tri(T['tri'], idx, level)
    point_ids = np.unique(T['simps'][tris].ravel())
    point_ids = [pt_id for pt_id in point_ids if pt_id not in src_ids]
    return np.asarray(point_ids)

def tri_shape_filter(train_map, query_rads, query_angles):
    reff = train_map['trads']/query_rads
    reffmean = np.mean(reff, axis=1)
    dangles = train_map['tangles'] - query_angles
    damean = np.mean(dangles, axis=1)

    dists = np.hstack((reff - reffmean.reshape(-1,1), dangles - damean.reshape(-1,1)))

    dists = np.linalg.norm(dists, axis=1)
    return np.argsort(dists), reffmean, damean