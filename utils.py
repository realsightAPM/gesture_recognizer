#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-03 20:02
# Last modified: 2017-07-26 20:32
# Filename: utils.py
# Description:
import os
import ast
import pickle
import logging
import sys
import re
import time
import math
import json

from logging import info

import cv2
import numpy as np
import scipy.misc

from PIL import Image
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

import features

from img_utils import get_pixels


fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
date_format = '%m %d %Y %H:%M:%S'


logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=date_format,
        filename='info.log')
#stream=sys.stdout)


def time_profile(precision=2):
    fmt = 'Time for >>{:^15s}<<: {:<.'+str(precision)+'f} s'
    def _time_profile(func):
        def _wrapper(*args, **kwargs):
            s = time.time()
            data = func(*args, **kwargs)
            e = time.time()
            print_log(fmt.format(func.__name__, e-s))
            return data
        return _wrapper
    return _time_profile


def read_binary_depth(fname, height=480, width=640):
    """
    Read 16 bit short from file.

    Author: David
    """
    data = np.fromfile(fname, dtype=np.int16)
    data = data.reshape((height, width))
    return data


def read_gray_depth(fname, height=480, width=640):
    """
    Read Grayscale depth from file.

    Author: David
    """
    im = scipy.misc.imread(fname, flatten=False)
    return im


def read_csv(fname):
    """
    Read csv file and parse into dictionary.

    The format of each line in CSV file should be like:
        key_0,val_00,val_01,val_02,...,val_0n
        key_1,val_10,val_11,val_12,...,val_1n
        ...

    Author: David
    """
    data = {}
    with open(fname, 'r') as f:
        for line in f:
            key, *vals = line.strip('\n').split(',')
            if len(vals) == 1:
                data[key] = ast.literal_eval(vals[0])
            else:
                data[key] = tuple(ast.literal_eval(x) for x in vals)
    return data


def read_rgb(fname):
    """
    Read RGB file pixels.

    Author: David
    """
    im = scipy.misc.imread(fname, flatten=False, mode='RGB')
    return im


def read_json(fname):
    with open(fname) as f:
        return json.load(f)


def read_lp_png(fname):
    img = Image.open(fname)
    return get_pixels(img)


def read(fname):
    """
    Invoke suitable function to read file contents.

    This is specialized, not for general purpose.

    Author: David
    """
    if fname.endswith('.bin'):
        return ('depth', read_binary_depth(fname))
    elif fname.endswith('.csv'):
        return ('lm', read_csv(fname))
    elif 'depth' in fname:
        return ('depth_png', read_rgb(fname))
    elif fname.endswith('.json'):
        return ('json', read_json(fname))
    elif fname.endswith('_l.png'):
        return ('leye', cv2.imread(fname, 0))
    elif fname.endswith('_r.png'):
        return ('reye', cv2.imread(fname, 0))
    elif fname.endswith('_lraw.png'):
        return ('lraw', cv2.imread(fname, 0))
    elif fname.endswith('_rraw.png'):
        return ('rraw', cv2.imread(fname, 0))
    else:
        return ('rgb', read_rgb(fname))


def set_dataset_cache(obj, fname, cache_path='caches'):
    """
    Save dataset with pickle.

    Author: David
    """
    if fname is None:
        return None
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    for key, val in obj.items():
        piece_name = '_'.join([fname, key]) + '.cache'
        path = os.path.join(cache_path, piece_name)
        with open(path, 'wb') as f:
            pickle.dump(val, f)


def get_dataset_cache(fname, users=None, cache_path='caches'):
    """
    Retrieve obj from file with pickle.

    Author: David
    """
    if fname is None:
        return None
    cache_files = [[re.split('_|\.', x), x] for x in os.listdir(cache_path)]
    if users is not None:
        cache_files = [x for x in cache_files if x[0][1] in users]
    data = {}
    for _info, _fname in cache_files:
        if _info[0] != fname:
            continue
        path = os.path.join(cache_path, _fname)
        with open(path, 'rb') as f:
            data[_info[1]] = pickle.load(f)
    return data



def set_cache(obj, fname, cache_path='caches'):
    """
    Cache normal objects with pickle.

    Author: David
    """
    if fname is None:
        return None
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    path = os.path.join(cache_path, fname) + '.cache'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_cache(fname, cache_path='caches'):
    """
    Retrieve normal objects from file with pickle.

    Author: David
    """
    if fname is None:
        return None
    path = os.path.join(cache_path, fname) + '.cache'
    print(path);
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = - axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rescale_value(value, vmin, vmax, rvmin, rvmax):
    """
    Rescale value from range [vmin, vmax] to new range [rvmin, rvmax].

    Author: David
    """
    if value is None:
        return 0
    sign = 1 if value > 0 else -1
    value = abs(value)
    rv = rvmin + (rvmax - rvmin) * (value - vmin) / (vmax - vmin)
    return rv 


def print_log(msg, *args, **kwargs):
    print(msg, *args, **kwargs)
    info(msg)


def select_features(x, features_selection):
    deletes = []
    if not (features_selection & features.F_T):
        deletes.extend(range(15, 25))
    if not (features_selection & features.F_E):
        deletes.extend(range(10, 15))
    if not (features_selection & features.F_D):
        deletes.extend(range(5, 10))
    if not (features_selection & features.F_A):
        deletes.extend(range(0, 5))
    return np.delete(x, deletes, axis=1)


def find_best_params(x, y, label='', score='accuracy'):
    X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2)

    tuned_parameters = {'estimator__gamma': [10**-i for i in range(5)], 'estimator__C': [10**i for i in range(4)]}

    print('-'*30)
    print("# Tuning hyper-parameters for %s\n" % label)
    model = OneVsOneClassifier(SVC())
    clf = GridSearchCV(model, tuned_parameters, cv=10,
            scoring=score, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Grid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred), '\n')
    params = {k.replace('estimator__', ''):v for k, v in clf.best_params_.items()}
    print('Best parameters ready to train:', params)
    print()
    return params


def pca_accuracy(x_train, y_train, x_test, y_test, start=100, end=60,
                 best_params=None):
    if best_params is None:
        best_params = {}
    res = []

    for p in range(start, end-1, -1):
        percent = p / 100
        if abs(percent - 1) < 1e-5:
            DECOMPOSITION = False
        else:
            DECOMPOSITION = True
        pca = PCA(percent)

        # Train set
        _train = x_train
        if DECOMPOSITION:
            _train = pca.fit_transform(_train)
        if res and _train.shape[1] == res[-1][1]:
            continue

        # Test set
        _test = x_test 
        if DECOMPOSITION:
            _test = pca.transform(_test)

        all_clf = OneVsOneClassifier(SVC(**best_params), n_jobs=-1)
        all_clf.fit(_train, y_train)

        h_combine = all_clf.predict(_test)
        score = accuracy_score(h_combine, y_test)

        _res = (percent, _test.shape[1], score)
        res.append(_res)
        print(('Variance retained: {:7.2%}, Dimensions retained: {:6d}'
               ', Accuracy: {:7.2%}'.format(*_res)))
    return res


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    depth_png = 'data/P5/G9/1_depth.png'
    depth_data = read_gray_depth(depth_png)
    plt.imshow(depth_data, cmap='gray')
    plt.show()
