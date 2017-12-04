#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-08 18:08
# Last modified: 2017-04-10 20:59
# Filename: img_utils.py
# Description:
import sys
from math import sqrt, ceil

import numpy as np
import matplotlib.pyplot as plt

from numpy import dot, arccos, mean, std
from numpy.linalg import norm, svd
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import butter, filtfilt


def get_pixels(img):
    pixels = np.array(img)
    mask = pixels > 0
    pixels = pixels[np.ix_(mask.any(1), mask.any(0))]
    fill_pixels = np.copy(pixels)
    m, n = pixels.shape
    for i in range(1, m-1):
        for j in range(1, n-1):
            white_cnt = np.count_nonzero(pixels[i-1:i+2, j-1:j+2])
            if pixels[i][j] == 0 and white_cnt >= 6:
                fill_pixels[i][j] = 255

    return fill_pixels


def find_center_of_img(pixels):
    center = [int(round(x)) for x in center_of_mass(pixels)]
    return center


def plot_img(pixels):
    center = find_center_of_img(pixels)

    plt.scatter(center[1], center[0], c='red')
    plt.imshow(pixels, cmap='gray')
    plt.xlabel('Width')
    plt.ylabel('Height')


def find_largest_circle(pixels, tpercent=0.8, silence=False):
    m, n = pixels.shape
    center = [int(round(x)) for x in center_of_mass(pixels)]
    r = 0
    for k in range(100):
        if not silence:
            sys.stdout.flush()
            sys.stdout.write('Iteration {}\r'.format(k+1))
        percent = 1
        while True:
            inside = 0
            cnt = 0
            inside_pixels = []
            for i in range(0, m):
                for j in range(0, n):
                    dist = sqrt((i-center[0])**2+(j-center[1])**2)
                    if dist > r:
                        continue
                    cnt += 1
                    if pixels[i][j] != 0:
                        inside += 1
                        inside_pixels.append([i, j])
            percent = inside / cnt
            if percent <= tpercent:
                break
            r += 1
        if not inside_pixels:
            continue
        inside_pixels = np.array(inside_pixels)
        t_center = np.average(inside_pixels, axis=0)
        if np.all(np.equal(t_center, center)):
            # print('\nReach largest possible circle during iteration {}'.format(k+1))
            break
        center = t_center
    sys.stdout.flush()
    return center, r


def plot_circle_on_img(pixels):
    center, r = find_largest_circle(pixels)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(pixels, cmap='gray')
    plt.scatter(center[1], center[0], c='red')
    circle = plt.Circle([center[1], center[0]], radius=r, color='r', fill=False, linewidth=1)
    ax.add_patch(circle)


def find_edge_of_img(pixels):
    edge_pixels = np.copy(pixels)
    m, n = pixels.shape
    for i in range(1, m-1):
        for j in range(1, n-1):
            if edge_pixels[i][j] == 0:
                continue
            white_cnt = np.count_nonzero(pixels[i-1:i+2, j-1:j+2])
            if white_cnt < 9:
                edge_pixels[i][j] = 255
            else:
                edge_pixels[i][j] = 0
    return edge_pixels


def plot_edge_of_img(pixels):
    m, n = pixels.shape
    center = find_center_of_img(pixels)
    edge_pixels = find_edge_of_img(pixels)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(edge_pixels, cmap='gray')
    plt.scatter(center[1], center[0], c='red')
    # circle = plt.Circle([center[1], center[0]], radius=r, color='r', fill=False, linewidth=1)
    # ax.add_patch(circle


def get_dists(edge_pixels, center, r, vmax=30):
    m, n = edge_pixels.shape
    dists = [[] for _ in range(360)]
    for i in range(0, m):
        for j in range(0, n):
            if edge_pixels[i][j] == 0:
                continue
            v = [i-center[0], j-center[1]]
            nv = norm(v)
            dist = nv - r
            if dist < 0:
                continue
            angle = arccos(dot(v, [1, 0]) / nv) * 180 / np.pi
            if v[1] > 0:
                angle = max(angle, 360 - angle)
            dists[int(ceil(angle)) % 360].append(dist)
    dists = [max(x) if x else 0 for x in dists]
    # mdist = max(dists)
    dists = [d / vmax for d in dists]
    return dists


def butter_lowpass_filtfilt(data, cutoff, order=5):
    b, a = butter(order, cutoff, btype='lowpass', analog=False)
    y = filtfilt(b, a, data)
    return y


def smooth_dists(dists, n=5):
    sdists = butter_lowpass_filtfilt(dists, 0.25, 10)
    sdists[sdists < 0] = 0
    return sdists


def plot_dists(edge_pixels, center, r, dists=None, sdists=None, **kwargs):
    cnt = 1
    cur = 1
    if dists is not None:
        cnt += 1
    if sdists is not None:
        cnt += 1
    if cnt == 1:
        raise ValueError("At least one dists should be provided")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(cnt, 1, cur)
    ax.imshow(edge_pixels, cmap='gray')
    plt.scatter(center[1], center[0], c='red')
    circle = plt.Circle([center[1], center[0]], radius=r, color='r', fill=False, linewidth=1)
    ax.add_patch(circle)
    ax.set_title('Largest possible circle to cut regions')
    cur += 1

    xlim = kwargs.get('xlim', (-180, 180))

    if dists is not None:
        ax = fig.add_subplot(cnt, 1, cur)
        ax.plot(range(-180, 180), dists)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1)
        ax.set_title('Distance series from different regions(un-filtered)')
        cur += 1

    if sdists is not None:
        ax = fig.add_subplot(cnt, 1, cur)
        ax.plot(range(-180, 180), sdists)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1)
        ax.set_title('Distance series from different regions(filtered)')


def get_sdists(pixels):
    edge_pixels = find_edge_of_img(pixels)
    center, r = find_largest_circle(pixels, silence=True)
    dists = get_dists(edge_pixels, center, r, vmax=40)
    sdists = smooth_dists(dists)
    return sdists


def difference(a, b):
    diff = abs(a-b)
    return diff


def correlation(a, b, normalize=True):
    if normalize:
        a = (a - mean(a)) / (std(a) * len(a))
        b = (b - mean(b)) /  std(b)
    return np.correlate(a, b)
