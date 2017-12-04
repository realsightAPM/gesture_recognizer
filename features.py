#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-06 10:58
# Last modified: 2017-04-21 17:54
# Filename: features.py
# Description:
import numpy as np
import cv2

from numpy.linalg import norm
from numpy import arccos, dot, clip, cross

from dataset import read_datasets, read_leap_motion_data
from utils import rescale_value


thres = [[-1e3, -40], [-40, -10], [-10, 10], [10, 40], [40, 1e3]]
F_A = 0b1
F_D = 0b10
F_E = 0b100
F_T = 0b1000
F_ALL = F_A | F_D| F_E | F_T


def clean_occupied(f):
    """
    Try to solve that one region would be occupied by more than one fingers.

    Author: David
    """
    if len(f[0]) > 1 and len(f[1]) == 0:
        k = f[0].pop(-1)
        f[1].append(k)
    for idx in range(1, 4):
        if len(f[idx]) <= 1:
            continue
        if len(f[idx-1]) == 0 and len(f[idx+1]) != 0:
            k = f[idx].pop(0)
            f[idx-1].append(k)
        elif len(f[idx-1]) != 0 and len(f[idx+1]) == 0:
            k = f[idx].pop(-1)
            f[idx+1].append(k)
        elif len(f[idx-1]) == 0 and len(f[idx+1]) == 0:
            diffl = abs(f[idx][0][0] - thres[idx][0])
            diffr = abs(f[idx][-1][0] - thres[idx][1])
            if diffl < diffr:
                k = f[idx].pop(0)
                f[idx-1].append(k)
            else:
                k = f[idx].pop(-1)
                f[idx+1].append(k)
    if len(f[-1]) > 1 and len(f[-2]) == 0:
        k = f[-1].pop(0)
        f[-2].append(k)

    # Re-ordered index
    indices = [-1 for k in range(5)]
    cf = []
    for region in f:
        if not region:
            cf.append(None)
        else:
            cf.append(max((v for v in region), key=lambda x: x[0]))
    for i in range(5):
        if cf[i] is None:
            continue
        val, idx = cf[i]
        indices[idx] = i
        cf[i] = val
    return cf, indices


def finger_angle_features(finger_pos, palm_pos, norm_vector, hand_dir):
    """
    Extract finger angle features.

    Calculate each angle of fingers with respect to hand_direction, based on
    clockwise or anticlockwise to determine the angle should be positive or
    negative and also which region should be in. In order to prevent the issue
    that more than one fingers would be in the same region, assign the nearest
    finger to the nearest adjacent region if not be occupied, otherwise select
    the maximum. Finally rescale the value to [0.5, 1] and use 0 to indicate
    no valid value.
    Author: David
    """
    angles = []
    ds = []
    ps = []
    fs = []
    for idx, f_pos in enumerate(finger_pos):
        f_dir = f_pos - palm_pos
        dist = f_dir.dot(norm_vector)
        f_proj = f_pos - dist * norm_vector
        f_proj_dir = f_proj - palm_pos

        norm_fproj = norm(f_proj_dir)
        norm_hand = norm(hand_dir)

        c = dot(f_proj_dir, hand_dir) / norm_fproj / norm_hand
        angle = arccos(clip(c, -1, 1)) * 180 / np.pi

        # vector direction
        angle_cross = cross(f_proj_dir, hand_dir) / norm_fproj / norm_hand
        direction = 1 if dot(angle_cross, norm_vector) < 0 else -1

        angles.append((direction * angle, idx))
        ds.append(direction)
        ps.append(f_proj_dir)
        fs.append(f_dir)

    angles.sort(key=lambda x: x[0])

    f = [[] for _ in range(5)]
    for angle_item in angles:
        for idx, (l_thres, h_thres) in enumerate(thres):
            if l_thres < angle_item[0] <= h_thres:
                f[idx].append(angle_item)
                break

    cf, idx = clean_occupied(f)
    cf = [rescale_value(v, 0, 90, 0.5, 1) for v in cf]
    return cf, fs, ps, ds, idx


def finger_distance_features(finger_pos, palm_pos, indices):
    dists = [0 for _ in range(5)]
    for idx, f_pos in enumerate(finger_pos):
        dist = norm(f_pos - palm_pos)
        dists[indices[idx]] = dist
    return dists


def finger_elevation_features(finger_dirs, proj_dirs, hand_dir, indices):
    eles = [0 for _ in range(5)]
    for idx in range(len(finger_dirs)):
        mv = finger_dirs[idx] - proj_dirs[idx]
        e = norm(mv)
        sign = 1 if dot(mv, hand_dir) > 0 else -1
        eles[indices[idx]] = e * sign
    return eles


def tips_distance_features(finger_pos, indices):
    dists = [0 for _ in range(10)]
    # dists = []
    size = len(finger_pos)
    for i in range(size):
        for j in range(i+1, size):
            dist = norm(finger_pos[i] - finger_pos[j])
            # dists.append(dist)
            k1 = indices[i]
            k2 = indices[j]
            k1, k2 = min(k1, k2), max(k1, k2)
            if k1 == 0:
                dists[k2-1] = dist
            elif k1 == 1:
                dists[k2+2] = dist
            elif k2 == 2:
                dists[k2+4] = dist
            else:
                dists[9] = dist
    # dists.extend(0 for _ in range(10-len(dists)))
    # dists.sort(key=lambda x: -x)
    return dists

def extract_features(lm_data):
    """
    Extract features from leap motion data.

    Author: David
    """
    f = []
    flatten_pos = np.array(lm_data['FingertipsPositions'])
    finger_pos = []
    fingers = int(lm_data['FingersNr'])
    for i in range(fingers):
        finger_pos.append(flatten_pos[i::5])
    finger_pos = np.array(finger_pos)
    palm_pos = np.array(lm_data['PalmPosition'])

    hand_dir = np.array(lm_data['HandDirection'])
    norm_vector = np.array(lm_data['PalmNormal'])
    finger_distances = np.array(lm_data['FingertipsDistances'])

    A, *rest = finger_angle_features(finger_pos,
                                     palm_pos, norm_vector, hand_dir)
    f.extend(A)
    
    fs = rest[0]
    ps = rest[1]
    idx = rest[3]

    D = finger_distance_features(finger_pos, palm_pos, idx)
    f.extend(D)

    E = finger_elevation_features(fs, ps, hand_dir, idx)
    f.extend(E)

    T = tips_distance_features(finger_pos, idx)
    f.extend(T)

    return f


def extract_lp_features(lm_data):
    """
    Extract features from our leap motion data.

    Author: David
    """
    f = []
    flatten_pos = np.array(lm_data['fingers_tip_position'])
    finger_pos = []
    fingers = int(lm_data['fingers_num'])
    for i in range(fingers):
        finger_pos.append(flatten_pos[i:i+3])
    finger_pos = np.array(finger_pos)
    palm_pos = np.array(lm_data['palm_position'])

    hand_dir = np.array(lm_data['hand_direction'])
    norm_vector = np.array(lm_data['palm_normal'])

    A, *rest = finger_angle_features(finger_pos,
                                     palm_pos, norm_vector, hand_dir)
    f.extend(A)

    idx = rest[3]
    D = finger_distance_features(finger_pos, palm_pos, idx)
    f.extend(D)

    fs = rest[0]
    ps = rest[1]
    idx = rest[3]
    E = finger_elevation_features(fs, ps, hand_dir, idx)
    f.extend(E)

    T = tips_distance_features(finger_pos, idx)
    f.extend(T)
    return f


def hog_single(img, cellx=20, celly=20):
    w, h = img.shape
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16
    bin = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = []
    mag_cells = []
    cellxn = w // cellx
    cellyn = h // celly
    for x in range(cellxn+1):
        for y in range(cellyn+1):
            bin_cells.append(bin[x*cellx:(x+1)*cellx, y*celly:(y+1)*celly])
            mag_cells.append(mag[x*cellx:(x+1)*cellx, y*celly:(y+1)*celly])
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps
    return np.float32(hist)


if __name__ == '__main__':
#    persons = read_datasets(k=1)
#    p = list(persons.keys())[0]
#    g = 'G5'
#    lm_data = persons[p][g][1]['lm']
#    f = extract_features(lm_data)
#    print(f)
#    # print(len(f))

    persons = read_leap_motion_data()
    p = list(persons.keys())[0]
    g = 'G5'
    lm_data = persons[p][g][1]['json']
    f = extract_lp_features(lm_data)
    print(f)
    print(len(f))
