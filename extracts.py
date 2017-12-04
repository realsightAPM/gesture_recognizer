#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-04 09:57
# Last modified: 2017-04-04 16:22
# Filename: extracts.py
# Description:
from logging import info


import matplotlib.pyplot as plt

from dataset import read_datasets


def extract_hand(rgb_data, depth_data, t_depth=10, t_size=20):
    fig = plt.subplot(121)
    fig.imshow(rgb_data)
    fig = plt.subplot(122)
    fig.imshow(depth_data, cmap='gray')
    plt.show()


def main():
    persons = read_datasets()
    rgb_data = persons['P7']['G9'][1]['rgb']
    depth_png = persons['P7']['G9'][1]['depth_png']
    extract_hand(rgb_data, depth_png)


if __name__ == '__main__':
    main()
