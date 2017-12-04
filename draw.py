#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-06 10:55
# Last modified: 2017-04-10 09:41
# Filename: draw.py
# Description:

import seaborn as sns


def draw_fingers(w, h, k, figs):
    g = 'G{}'.format(k)
    ax = fig.add_subplot(w, h, k, projection='3d')

    for i in range(1, 11):
        d = persons[p][g][i]['lm']
        fingers = int(d['FingersNr'])
        if fingers == figs[k-1]:
            break
    flatten_pos = np.array(d['FingertipsPositions'])
    finger_pos = []
    angles = []
    for i in range(fingers):
        finger_pos.append(flatten_pos[i::5])
    finger_pos = np.array(finger_pos)
    # print('Fingers: {}'.format(len(finger_pos)))
    # print('Fingers Pos:\n', finger_pos)
    palm_pos = np.array(d['PalmPosition'])
    # print('Palm Pos:\n', palm_pos)

    ax.set_xlim([palm_pos[0] - 100, palm_pos[0] + 100])
    ax.set_ylim([palm_pos[1] - 50, palm_pos[1] + 50])
    ax.set_zlim([palm_pos[2] - 50, palm_pos[2] + 50])
    ax.scatter(palm_pos[0], palm_pos[1], palm_pos[2], zdir='z', c='black', s=64)

    hand_direction = np.array(d['HandDirection'])
    # print('Hand Direction:\n', hand_direction)
    norm_vector = np.array(d['PalmNormal'])
    # print('Normal Vector:\n', norm_vector)

    norm_end = 50 * norm_vector + palm_pos
    ax.plot([palm_pos[0], norm_end[0]], [palm_pos[1], norm_end[1]], [palm_pos[2], norm_end[2]], c='green')
    ax.text(*norm_end, 'NV')
    hand_end = 150 * hand_direction + palm_pos
    ax.plot([palm_pos[0], hand_end[0]], [palm_pos[1], hand_end[1]], [palm_pos[2], hand_end[2]], c='green')
    ax.text(*hand_end, 'HD')

    angles = []
    for f_pos in finger_pos:
        f_direction = f_pos - palm_pos
        dist = f_direction.dot(norm_vector)
        f_proj = f_pos - dist * norm_vector
        # Plot f_pos
        ax.scatter(f_pos[0], f_pos[1], f_pos[2], c='red', s=64)
        # Plot f_proj
        ax.scatter(f_proj[0], f_proj[1], f_proj[2], c='blue', s=64)
        # Line f_pos and f_proj
        ax.plot([f_pos[0], f_proj[0]], [f_pos[1], f_proj[1]], [f_pos[2], f_proj[2]], c='red')
        # Line f_proj and palm_pos
        ax.plot([f_proj[0], palm_pos[0]], [f_proj[1], palm_pos[1]], [f_proj[2], palm_pos[2]], c='blue')
        c = dot(f_direction, hand_direction) / norm(f_direction) / norm(hand_direction)
        angle = arccos(clip(c, -1, 1)) * 180 / np.pi
        refined_angle = angle / 180 + 0.5
        angle_cross = cross(f_direction, hand_direction) / norm(f_direction) / norm(hand_direction)
        direction = 1 if -dot(angle_cross, norm_vector) > 0 else -1
        # print('Angle of {}:\n [Degree]: {:.2f} [Scale]: {:.3f}'.format(f_pos, angle, refined_angle))
        angles.append(direction * angle)

    draw_finger_regions(ax, palm_pos, hand_end, norm_vector)


# Draw finger regions
def draw_finger_regions(ax, palm_pos, hand_end, norm_vector):
    region_angles = [40, 10, -10, -40]  # anticlockwise angle
    regions = []
    for idx, angle in enumerate(region_angles):
        finger_region = np.dot(hand_end, rotation_matrix(norm_vector,  angle / 360 * np.pi))
        ax.plot([palm_pos[0], finger_region[0]],
                [palm_pos[1], finger_region[1]],
                [palm_pos[2], finger_region[2]],
                c='black')
        regions.append(finger_region)
        ax.text(*finger_region, '{}'.format(-angle))


# Draw normal plane
def draw_norm_plane(ax):
    d = palm_pos.dot(norm_vector)
    x_range = range(int(palm_pos[0])-100, int(palm_pos[0])+100, 5)
    y_range = range(int(palm_pos[1])-100, int(palm_pos[1])+100, 5)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = (d - norm_vector[0] * xx - norm_vector[1] * yy) * 1. / norm_vector[2]
    ax.plot_surface(xx, yy, zz, color='white')


def plot(*args, **kwargs):
    sns.distplot(*args, **kwargs, hist=False)
