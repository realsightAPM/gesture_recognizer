#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-11 10:46
# Last modified: 2017-06-02 09:06
# Filename: recognition.py
# Description:
import json
import time

from random import sample

import numpy as np
import xgboost as xgb
import pandas as pd

from sklearn.metrics import accuracy_score

import features

from features import extract_lp_features, hog_single
from utils import get_cache, read, time_profile
from dataset import read_leap_motion_data


MAX_DIST = 340.42
#clf = get_cache('prfediction_model')
clf = xgb.Booster()
clf.load_model('caches/combined_model')
# features_selection = features.F_A | features.F_D | features.F_T
features_selection = features.F_ALL
deletes = []
if not (features_selection & features.F_T):
    deletes.extend(range(15, 25))
if not (features_selection & features.F_E):
    deletes.extend(range(10, 15))
if not (features_selection & features.F_D):
    deletes.extend(range(5, 10))
if not (features_selection & features.F_A):
    deletes.extend(range(0, 5))  


def extract_feature_from_json(jd):
    f = np.array(extract_lp_features(jd))
    f[5:] = f[5:] / MAX_DIST
    f = np.delete([f], deletes, axis=1)
    return f


def extract_feature_from_img(limg, rimg):
    hog_l = np.array(hog_single(limg)).reshape((1, -1))
    hog_r = np.array(hog_single(rimg)).reshape((1, -1))
    return np.append(hog_l, hog_r, axis=1)


def _predict(features):
    
    _f = xgb.DMatrix(features)
    h = clf.predict(_f)
    return int(h[0])


def _extract_xy_pos(jd):
    x, h, y = jd['palm_position']
    y = -y
    return (x, y, h)


def predict(jd, limg, rimg):
    num_features = extract_feature_from_json(jd)
    hog_features = extract_feature_from_img(limg, rimg)

    features = np.append(num_features, hog_features, axis=1)
    h = _predict(features)
    pos = _extract_xy_pos(jd)
    return (h, pos)


def main():
    fnames = []
    for p in range(1, 14):
        for g in range(1, 11):
            for r in range(1, 21):
                pgr = 'LP_data/dataset/P{}/G{}/R{}'.format(p, g, r)
                json_file = pgr + '.json'
                limg_file = pgr + '_l.png'
                rimg_file = pgr + '_r.png'
                fnames.append((json_file, limg_file, rimg_file, g))
    for i in range(50):
        hs = []
        gs = []
        cost = 0
        for json_file, limg_file, rimg_file, g in sample(fnames, 200):
            _, jd = read(json_file)
            _, limg = read(limg_file)
            _, rimg = read(rimg_file)
            start = time.time()
            h = predict(jd, limg, rimg)
            end = time.time()
            cost += end-start
            hs.append(h)
            gs.append(g-1)
        per_cost = cost / 200
        print('Per prediction cost: {:.4f}'.format(per_cost))
        print('Predictions per sec: {:.2f}'.format(1 / per_cost))
        print('accuracy: {:.2%}'.format(accuracy_score(hs, gs)))


if __name__ == '__main__':
    main()
