import cv2
import numpy as np
from scipy.optimize import fsolve, linear_sum_assignment
from scipy.spatial import Delaunay


# data to remap image pixels to lat long
vlon = np.asarray([[80, 52.495058], 
                   [809, 52.492678], 
                   [1271, 52.491173],]).T
ulat = np.asarray([[15, 13.245452], 
                   [518, 13.248172], 
                   [1003, 13.250772],]).T

VSLOPE, VINTER = np.polyfit(vlon[0], vlon[1], deg=1)
USLOPE, UINTER = np.polyfit(ulat[0], ulat[1], deg=1)
ZSAT = 150

def rearrange(arr, start_idx):
    return np.hstack((arr[start_idx:], arr[:start_idx]))

def contour_map(cnts, azimuth):
    # convert contours to polar coordinates
    T = {
        "centers": [],
        "areas": [],
        "simps": [],
        "tri": 0
    }

    for cnt in cnts:
        M = cv2.moments(cnt)
        center = M['m10']/M['m00'], M['m01']/M['m00']
        T['centers'].append(center)
        T['areas'].append(cv2.contourArea(cnt))

    T['centers'] = np.asarray(T['centers']).astype(np.float32) # type: ignore
    T['areas'] = np.asarray(T['areas']) # type: ignore

    # use delaunay to find neighbor objects
    T["tri"] = Delaunay(T['centers'])

    # order simp vertices according to angle in polar coordinates 
    T['simps'] = T['tri'].simplices
    n_simps = len(T['simps'])

    T['tcenter'] = np.zeros((n_simps, 2), dtype=np.float32)
    T['tarea'] = np.zeros((n_simps), dtype=np.float32)
    T['tangles'] = np.zeros((n_simps, 3), dtype=np.float32)
    T['trads'] = np.zeros((n_simps, 3), dtype=np.float32)
    for ii, simp in enumerate(T['simps']):
        # find tri vertices coordinates and estimate its center position and area 
        centers = T['centers'][simp]
        T['tarea'][ii] = cv2.contourArea(centers.reshape(-1,1,2)) 
        T['tcenter'][ii] = np.mean(centers, axis=0)

        # go to tri center to recalculate vertices in polar coordinates
        centers -= T['tcenter'][ii]
        trads = np.linalg.norm(centers, axis=1) 
        tangles = np.arctan2(centers[:,0], centers[:,1]) - azimuth

        # rearrange vertices from highest to lowest angle  
        tangles = np.where(tangles<0, 2*np.pi+tangles, tangles)
        max_tangle_id = np.argmax(tangles)
        T['simps'][ii] = rearrange(simp, max_tangle_id)
        T['trads'][ii] = rearrange(trads, max_tangle_id)
        T['tangles'][ii] = rearrange(tangles, max_tangle_id)
    return T

def mask2cnts(mask):
    q_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(q_mask, 1, 2)
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt)>=64]
    return cnts

def mad_cost(pts0, pts1, tareas):
    cost_matrix = np.zeros((len(pts0), len(pts1), 2), dtype=np.float32)
    for ii, pt in enumerate(pts0):
        cost_matrix[ii] = pts1 - pt

    cost_matrix = np.linalg.norm(cost_matrix, axis=2)
    rows, cols = linear_sum_assignment(cost_matrix)
    norm_cost = cost_matrix[rows, cols]/tareas[cols]**0.5
    return norm_cost, rows, cols


def to_homo(points):
    return np.hstack((points, np.ones((len(points),1), dtype=np.float32)))

def lstsq_affine_matrix(src_pts, dst_pts):
    A = src_pts.copy()
    b = dst_pts.copy()
    A = np.hstack((A, A)).reshape(-1,2)
    A = to_homo(A)
    A = np.hstack((A, np.zeros(A.shape)))
    for ii, a in enumerate(A):
        if ii%2:
            A[ii] = np.hstack((a[3:], a[:3]))

    b = b.ravel().reshape(-1,1)
    return np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1,3)


def get_iou_cost(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection)/np.sum(union)


def affine2rot(x, affine_matrix):
    a11, a12 = affine_matrix[0, :2]
    a21, a22 = affine_matrix[1, :2]
    
    return (a11**2 + a12**2 + x[0]**2 - x[2]**2,
            a21**2 + a22**2 + x[1]**2 - x[2]**2,
            a11*a21 + a12*a22 + x[0]*x[1])

def find_rot_matrix(affine_matrix):
    x =  fsolve(affine2rot, [1, 1, 1], args=(affine_matrix))
    RotM = np.eye(3)

    RotM[:2, :2] = affine_matrix[:2, :2]
    RotM[:2, 2] = x[:2]
    RotM /= x[2]
    RotM[2] = np.cross(RotM[0], RotM[1])
    return RotM, x[2]

def get_gps_data(im_w, im_h, AfM, affine_matrix_inv1):
    q_cnt = np.asarray([[0, 0], [im_w, 0], [im_w, im_h], [0, im_h]])
    sat_cnt = to_homo(q_cnt).dot(affine_matrix_inv1.T).astype(np.float32).reshape((-1,1,2))
    RotM, scale = find_rot_matrix(AfM)
    sat_center = np.mean(sat_cnt.reshape(-1,2), axis=0)

    # drone world coordinates
    lat = VSLOPE * sat_center[1] + VINTER
    lon = USLOPE * sat_center[0] + UINTER
    height = RotM[2, 2] * scale * ZSAT
    return sat_center, (lat, lon, height)