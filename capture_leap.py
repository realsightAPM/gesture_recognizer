#!/usr/local/bin/python2
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-06 16:28
# Last modified: 2017-04-08 18:28
# Filename: plot_leap.py
# Description:
import sys, threading, json, time, os, ctypes

import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib import animation


import Leap


def convert_distortion_maps(image):

    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length/2, dtype=np.float32)
    ymap = np.zeros(distortion_length/2, dtype=np.float32)

    for i in range(0, distortion_length, 2):
        xmap[distortion_length/2 - i/2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length/2 - i/2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width/2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width/2))

    #resize the distortion map to equal desired destination image size
    resized_xmap = cv2.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)
    resized_ymap = cv2.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)

    #Use faster fixed point maps
    coordinate_map, interpolation_coefficients = cv2.convertMaps(
        resized_xmap, resized_ymap, cv2.CV_32FC1, nninterpolation = False)

    return coordinate_map, interpolation_coefficients


def undistort(image, coordinate_map, coefficient_map, width, height):
    destination = np.empty((width, height), dtype = np.ubyte)

    #wrap image data in numpy array
    i_address = int(image.data_pointer)
    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    # as ctypes array
    as_ctype_array = ctype_array_def.from_address(i_address)
    # as numpy array
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    #remap image to destination
    destination = cv2.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation = cv2.INTER_LINEAR)

    #resize output to desired destination size
    destination = cv2.resize(destination,
                             (width, height),
                             0, 0,
                             cv2.INTER_LINEAR)
    return destination


class SampleListener(Leap.Listener):
    def __init__(self, *args, **kwargs):
        super(SampleListener, self).__init__(*args, **kwargs)
        self.map_initialized = False

    def on_frame(self, controller):
        self.data = {}
        data = {}
        frame = controller.frame()
        if not frame.is_valid:
            return
        if not self.map_initialized:
            self.lcoord, self.lcoef = convert_distortion_maps(
                frame.images[0])
            self.rcoord, self.rcoef = convert_distortion_maps(
                frame.images[1])
            self.map_initialized = True
        image = frame.images[0]
        limage = Image.fromarray(
            undistort(image, self.lcoord, self.lcoef, 400, 400))
        image = frame.images[1]
        rimage = Image.fromarray(
            undistort(image, self.rcoord, self.rcoef, 400, 400))
        self.pil_limage = limage 
        self.pil_rimage = rimage


def main():

    listener = SampleListener()
    controller = Leap.Controller()
    controller.set_policy(Leap.Controller.POLICY_IMAGES)
    controller.add_listener(listener)
    time.sleep(1)

    def filter_image(im):
        a = np.array(listener.pil_limage)
        thres = 90
        a[a < thres] = 0
        a[a >= thres] = 255
        return a

    # fig = plt.figure(figsize=(16, 12))
    img = listener.pil_limage
    pixels = filter_image(img)
    img = Image.fromarray(pixels)
    cnt = len([x for x in os.listdir('LP_data') if x.startswith('frame')])
    img.save('LP_data/frame_{}.png'.format(cnt+1))
    # ax = plt.imshow(pixels, cmap='gray')

    def animate(i):
        a = filter_image(listener.pil_limage)
        ax.set_array(a)
        return [ax]

    # print 'Press Enter to quit...'

    try:
        pass
        # anim = animation.FuncAnimation(fig, animate, interval=100)
        # plt.show()
        # sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        controller.remove_listener(listener)

if __name__ == '__main__':
    main()
