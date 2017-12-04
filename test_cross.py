#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-05 16:54
# Last modified: 2017-04-06 16:04
# Filename: test_cross.py
# Description:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy import array, dot, cross, arccos, clip
from numpy.linalg import norm


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')



def line_to(vec_to, vec_from=[0, 0, 0]):
    ax.plot([vec_from[0], vec_to[0]],
            [vec_from[1], vec_to[1]],
            [vec_from[2], vec_to[2]])

base_vec = [1, 1, 0]
vec1 = [1, 0.5, 0]
vec2 = [0.5, 1, 0]
vec3 = [-1, 1, 0]
vec4 = [-1, -1, 0]
vec5 = [1, 0, 0]
vec6 = [1, -1, 0]

norm_vec = [0, 0, 1]


def clockwise(vec, target, norm_vec):
    c = cross(vec, target) / norm(vec) / norm(target)
    d = dot(c, norm_vec)
    if d > 0:
        return 'Pos'
    elif d < 0:
        return 'Neg'
    else:
        return 'Zero'

# vecs = [vec1, vec2, vec3, vec4, vec5, vec6]
vecs = []
for i in range(-10, 10, 2):
    for j in range(-i, 10, 2):
        vecs.append([i / 10, j / 10, 0])

ax.text(*base_vec, 'Base')
for vec in vecs:
    res = clockwise(vec, base_vec, norm_vec)
    line_to(vec)
    ax.text(*vec, res)

plt.show()
