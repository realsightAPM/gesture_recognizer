#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-10 09:34
# Last modified: 2017-04-21 11:08
# Filename: train.py
# Description:
import sys

from collections import defaultdict
from logging import warning

import numpy as np
import xgboost as xgb

from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image

import features

from utils import print_log, select_features
from img_utils import get_pixels, find_center_of_img, plot_img
from img_utils import find_largest_circle, plot_circle_on_img
from img_utils import find_edge_of_img, plot_edge_of_img, get_dists
from img_utils import smooth_dists, plot_dists, difference, correlation
from img_utils import get_sdists

#gs = []
#for i in range(1, 11):
#    print_log('Processing frame_{}.png'.format(i))
#    img = Image.open("LP_data/frame_{}.png".format(i))
#    pixels = get_pixels(img)
#    edge_pixels = find_edge_of_img(pixels)
#    center, r = find_largest_circle(pixels)
#    dists = get_dists(edge_pixels, center, r, vmax=40)
#    sdists = smooth_dists(dists)
#    gs.append(sdists[80:280])
#print_log('Gesture Template processing finished.')

def predict_img(pixels, ret_corr=False):
    sdists = get_sdists(pixels)[80:280]
    best_corr = 0
    best_gesture = -1
    for t in range(10):
        for k in range(-20, 20):
            roll_sdists = np.roll(gs[t], k)
            try:
                corr = correlation(sdists, roll_sdists)
            except Exception as e:
                warning(e)
                corr = 0
            if corr > best_corr:
                best_corr = corr
                best_gesture = t
    if ret_corr:
        return best_gesture, best_corr
    else:
        return best_gesture


def precision_info(precision_dict, iterations, title=None):
    print('\n', end='')
    if title:
        print_log(title, sep='')
    print_log('After {:3d} iterations:'.format(iterations))
    for key, data in precision_dict.items():
        print_log('-------')
        print_log('{:^20s}'.format(key))
        print_log('Max precision for {}: {:6.2%}'.format(key, max(p for p in data)))
        print_log('Min precision for {}: {:6.2%}'.format(key, min(p for p in data)))
        print_log('Avg precision for {}: {:6.2%}'.format(key, sum(p for p in data) / len(data)))
        print_log('-------')


def precision_for(raw_A, y, features_selection, test_size=0.1, ITER_TIMES=50):
    precisions = defaultdict(list)
    fmt = 'Precision for {:10s}'
    fs = ['A', 'D', 'E', 'T']
    deletes = []
    A = raw_A
    if not (features_selection & features.F_T):
        deletes.extend(range(15, 25))
        fs.remove('T')
    if not (features_selection & features.F_E):
        deletes.extend(range(10, 15))
        fs.remove('E')
    if not (features_selection & features.F_D):
        deletes.extend(range(5, 10))
        fs.remove('D')
    if not (features_selection & features.F_A):
        deletes.extend(range(0, 5))
        fs.remove('A')
    A = np.delete(A, deletes, axis=1)
    title = fmt.format('+'.join(fs))

    for r in range(ITER_TIMES):
        ms = []
        m = 'Round {:3d}: '.format(r+1)
        ms.append(m)
        A_train, A_test, y_train, y_test = train_test_split(A, y, test_size=test_size)

        dtrain = xgb.DMatrix(A_train, label=y_train)
        dtest = xgb.DMatrix(A_test, label=y_test)
        param = {'silent': 1, 'objective': 'multi:softmax', 'num_class': 10}
        num_round = 50
        bst = xgb.train(param, dtrain, num_round)
        h = np.array(bst.predict(dtest)).astype(int)
        p = accuracy_score(h, y_test)
        m = 'XGBoost precision: {:5.2%}'.format(p)
        ms.append(m)
        precisions['XGBoost'].append(p)

        ms.append('\r')
        sys.stdout.flush()
        sys.stdout.write(' '.join(ms))

    precision_info(precisions, ITER_TIMES, title)
    return precisions


def feature_precisions(A, y, feature_sels, test_size=0.2, ITER_TIMES=50, classifier='xgboost', params=None):
    precisions = defaultdict(list)
    xgb_params = {'silent': 1, 'objective': 'multi:softmax', 'num_class': 10}
    if params is not None and classifier == 'xgboost':
        xgb_params.update(params)
    elif params is None:
        params = {}
    num_round = 50
    for r in range(ITER_TIMES):
        A_train, A_test, y_train, y_test = train_test_split(A, y, test_size=test_size)
        ms = []
        for key, features_selection in feature_sels.items():
            _A_train = select_features(A_train, features_selection)
            _A_test = select_features(A_test, features_selection)
            
            if classifier == 'xgboost':
                _A_train = xgb.DMatrix(_A_train, label=y_train)
                _A_test = xgb.DMatrix(_A_test, label=y_test)
                clf = xgb.train(xgb_params, _A_train, num_round)
            elif classifier == 'svm':
                clf = OneVsOneClassifier(SVC(**params))
                clf.fit(_A_train, y_train)
            h = np.array(clf.predict(_A_test)).astype(int)
            p = accuracy_score(h, y_test)

            precisions[key].append(p)
            ms.append('{:>7s} precision:{:7.2%}'.format(key, p))
        sys.stdout.flush()
        sys.stdout.write('Round {:3d}/{:3d}:{}\r'.format(r+1, ITER_TIMES, '|'.join(ms)))

    precision_info(precisions, ITER_TIMES)
    return precisions


def train_clf(A, y, features_selection, test_size=0.2, ITER_TIMES=50,
              num_round=200, params=None, classifier='xgboost'):
    best_clf = None
    best_accuracy = 0
    
    if params is None:
        params = {}
        
    deletes = []
    if not (features_selection & features.F_T):
        deletes.extend(range(15, 25))
    if not (features_selection & features.F_E):
        deletes.extend(range(10, 15))
    if not (features_selection & features.F_D):
        deletes.extend(range(5, 10))
    if not (features_selection & features.F_A):
        deletes.extend(range(0, 5))
    A = np.delete(A, deletes, axis=1)

    fmt = 'Round {:3d}/{:3d}: Current accuracy {:6.2%}, Best accuracy {:6.2%}\r'
    for r in range(ITER_TIMES):
        A_train, A_test, y_train, y_test = train_test_split(A, y, test_size=test_size)
        if classifier == 'xgboost': 
            A_train = xgb.DMatrix(A_train, label=y_train)
            A_test = xgb.DMatrix(A_test, label=y_test)
            xgb_params = {'max_depth': 3, 'eta': 0.1, 'silent': 1,
                          'objective': 'multi:softmax', 'num_class': 10}
            xgb_params.update(params)
            clf = xgb.train(xgb_params, A_train, num_round)
        elif classifier == 'svm':
            clf = OneVsOneClassifier(SVC(**params))
            clf.fit(A_train, y_train)
        h = np.array(clf.predict(A_test)).astype(int)
        p = accuracy_score(h, y_test)
        if p > best_accuracy:
            best_accuracy = p
            best_clf = clf

        sys.stdout.flush()
        sys.stdout.write(fmt.format(r+1, ITER_TIMES, p, best_accuracy))
    return best_clf
