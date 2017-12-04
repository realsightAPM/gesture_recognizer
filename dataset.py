#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-03 20:47
# Last modified: 2017-04-11 19:51
# Filename: dataset.py
# Description:
import os
import random
import math
from pprint import pprint

from concurrent.futures import ThreadPoolExecutor
from logging import info, error

from collections import defaultdict
from functools import partial

from utils import read, get_dataset_cache, set_dataset_cache, time_profile
from utils import get_cache, set_cache, print_log


base_dir = 'data'


@time_profile()
def read_leap_motion_data(base_dir=base_dir, load_cache=True,
                          cache_name='leap_motion_data', workers=8):
    """
    Load leap motion data with ThreadPoolExecutor, use cache if it exists.

    Return:
        persons: A dictionary. persons[PERSON_ID][GESTURE_ID][ROUND_ID]
            'fingers_dis_direction': list(list(float))
            'fingers_dis_position': list(list(float))
            'fingers_int_direction': list(list(float))
            'fingers_int_position': list(list(float))
            'fingers_num': int
            'fingers_pro_direction': list(list(float))
            'fingers_pro_position': list(list(float)
            'fingers_tip_direction': list(list(float))
            'fingers_tip_position': list(list(float))
            'hand_confidence': float
            'hand_direction': list(float)
            'hand_sphere_center': list(float)
            'hand_sphere_radius': float
            'palm_normal': list(float)
            'palm_position': list(float)
            'palm_position_refined': list(float)
    Author: David
    """
    files = []
    users = set()
    for root, _, _files in os.walk(base_dir):
        if len(_files) <= 1:
            continue
        print(root)
        _, person, gesture = root.rsplit('\\', 2)
        files.append([person, gesture, _files])
        users.add(person)

    if load_cache:
        print_log('Trying fetch cache for persons')
        persons = get_cache(cache_name)
        if persons is None:
            print_log('No cache, first building')
            persons = defaultdict(partial(defaultdict,
                                  partial(defaultdict, defaultdict)))
        elif len(persons) == len(users):
            print_log('Using cache for persons')
            return persons
        else:
            print_log('Some cache is missing, rebuilding')
    else:
        persons = defaultdict(partial(defaultdict,
                              partial(defaultdict, defaultdict)))

    _diff = users - set(persons.keys())
    files = [k for k in files if k[0] in _diff]
    errors = False

    def _read_datasets(dataset, target):
        nonlocal errors
        for person, gesture, files in dataset:
            print_log('Processing {:<3s} - {:<3s}'.format(person, gesture))
            for f in files:
                try:
                    e = f.find('_')
                    if e == -1:
                        e = f.find('.')
                    idx = int(f[1:e])
                    path = os.path.join(base_dir, person, gesture, f)
                    key, val = read(path)
                    target[person][gesture][idx][key] = val
                except Exception as e:
                    error(e)
                    errors = True
                    return

    chunk_size = int(math.ceil(len(files) / workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx in range(workers+1):
            chunk = files[idx * chunk_size:(idx+1) * chunk_size]
            executor.submit(_read_datasets, chunk, persons)
    print_log('Setting cache for persons')
    if not errors:
        set_cache(persons, cache_name)
    return persons


@time_profile()
def read_datasets(base_dir=base_dir, load_cache=True,
                  cache_name='persons', workers=8, k=0, **kwargs):
    """
    Load datasets with ThreadPoolExecutor, use cache if it exists.

    Return:
        persons: A dictionary. persons[PERSON_ID][GESTURE_ID]
            'depth': short 16 bits
            'lm': leap motion args
            'depth_png': rgb for depth image
            'rgb': original rgb image
    Author: David
    """
    files = []
    users = set()
    for root, _, _files in os.walk(base_dir):
        if len(_files) <= 1:
            continue
        _, person, gesture = root.split('/')
        files.append([person, gesture, _files])
        users.add(person)
    _users = kwargs.get('users', None)
    if _users:
        users = _users
    elif k > 0:
        users = random.sample(users, k)

    if load_cache:
        persons = get_dataset_cache(cache_name, users)
        if len(persons) == len(users):
            print_log('Using cache for persons')
            return persons
        else:
            print_log('Some cache is missing, rebuilding')
    else:
        persons = []


    files = [f for f in files if f[0] in users if f[0] not in persons]
    device = kwargs.get('device', '')
    if device == 'leap motion':
        for item in files:
            item[2] = [f for f in item[2] if f.endswith('.csv')]
    elif device == 'kinect':
            item[2] = [f for f in item[2] if not f.endswith('.csv')]


    def _read_datasets(dataset, target):
        for person, gesture, files in dataset:
            print_log('Processing {:<3s} - {:<3s}'.format(person, gesture))
            for f in files:
                idx = int(f[:f.find('_')])
                path = os.path.join(base_dir, person, gesture, f)
                key, val = read(path)
                target[person][gesture][idx][key] = val

    persons = defaultdict(partial(defaultdict,
                          partial(defaultdict, defaultdict)))
    chunk_size = int(math.ceil(len(files) / workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx in range(workers+1):
            chunk = files[idx * chunk_size:(idx+1) * chunk_size]
            executor.submit(_read_datasets, chunk, persons)
    print_log('Setting cache for persons')
    if not device:
        set_dataset_cache(persons, cache_name)
    return persons


if __name__ == '__main__':
    # persons = read_datasets(base_dir, load_cache=False, device='leap motion')
    print(read_leap_motion_data('LP_data/dataset').keys())
