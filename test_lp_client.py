#!/usr/local/bin/python2
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-06 16:28
# Last modified: 2017-05-28 16:18
# Filename: test_lp_client.py
# Description:
import sys
import json
import time
import ctypes
import socket
import zlib
import pickle
import optparse

from collections import deque, Counter

import numpy as np
import cv2
import Leap


pos_key_idx = {
    1: 'fingers_pro_position',
    2: 'fingers_int_position',
    3: 'fingers_dis_position'}
direction_key_idx = {
    1: 'fingers_pro_direction',
    2: 'fingers_int_direction',
    3: 'fingers_dis_direction'}


def parse_args():

    parser = optparse.OptionParser()

    help_msg = "The port to connected to. Default is 9999."
    parser.add_option('--port', type='int', default=9999, help=help_msg)

    help_msg = "The host to connected to. Default is 127.0.0.1"
    parser.add_option('--host', help=help_msg, default='127.0.0.1')

    options, args = parser.parse_args()

    if len(args) != 0:
        parser.error('Invalid Arguments')

    return options


def convert_distortion_maps(image):
    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length/2, dtype=np.float32)
    ymap = np.zeros(distortion_length/2, dtype=np.float32)

    for i in range(0, distortion_length, 2):
        xmap[distortion_length/2 - i/2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length/2 - i/2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width/2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width/2))

    resized_xmap = cv2.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)
    resized_ymap = cv2.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)

    coordinate_map, interpolation_coefficients = cv2.convertMaps(
        resized_xmap, resized_ymap, cv2.CV_32FC1, nninterpolation=False)

    return coordinate_map, interpolation_coefficients


def undistort(image, coordinate_map, coefficient_map, width, height):
    destination = np.empty((width, height), dtype=np.ubyte)

    i_address = int(image.data_pointer)
    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    as_ctype_array = ctype_array_def.from_address(i_address)
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    destination = cv2.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation=cv2.INTER_LINEAR)

    destination = cv2.resize(destination,
                             (width, height),
                             0, 0,
                             cv2.INTER_LINEAR)
    return destination


def filter_image(im):
    a = np.array(im)
    thres = 90
    a[a < thres] = 0
    a[a >= thres] = 255
    return a


def dround(l, p=5):
    if isinstance(l, list):
        return map(lambda x: round(x, p), l)
    else:
        return round(l, p)


class SampleListener(Leap.Listener):
    def __init__(self, *args, **kwargs):
        super(SampleListener, self).__init__(*args, **kwargs)
        self.map_initialized = False

    def get_data(self):
        limage = getattr(self, 'pil_limage', None)
        rimage = getattr(self, 'pil_rimage', None)
        jd = getattr(self, 'jd_data', None)
        if jd is None or limage is None or rimage is None:
            return None
        limage = filter_image(limage).tostring()
        rimage = filter_image(rimage).tostring()
        return (jd, limage, rimage)

    def on_frame(self, controller):
        data = {}
        frame = controller.frame()
        hand = frame.hands[0]
        if not frame.is_valid or not hand.is_valid:
            self.jd_data = None
            return
        data['hand_direction'] = dround(hand.direction.to_float_array())
        data['hand_confidence'] = dround(hand.confidence)
        data['hand_sphere_center'] = dround(hand.sphere_center.to_float_array())
        data['hand_sphere_radius'] = dround(hand.sphere_radius)

        data['palm_normal'] = dround(hand.palm_normal.to_float_array())
        data['palm_position'] = dround(hand.palm_position.to_float_array())
        data['palm_position_refined'] = dround(hand.stabilized_palm_position.to_float_array())

        # Rotation ?
        # Scaling ?
        # Translation ?
        fingers = hand.fingers.extended()
        data['fingers_num'] = len(fingers)
        # Skip metacarpal since thumb does not have one
        data['fingers_pro_position'] = []
        data['fingers_int_position'] = []
        data['fingers_dis_position'] = []
        data['fingers_tip_position'] = []
        data['fingers_pro_direction'] = []
        data['fingers_int_direction'] = []
        data['fingers_dis_direction'] = []
        data['fingers_tip_direction'] = []

        for finger in fingers:
            data['fingers_tip_position'].extend(
                dround(finger.tip_position.to_float_array()))
            data['fingers_tip_direction'].extend(
                dround(finger.direction.to_float_array()))
            for bone_idx in range(1, 4):
                bone = finger.bone(bone_idx)
                data[pos_key_idx[bone_idx]].extend(
                    dround(bone.center.to_float_array()))
                data[direction_key_idx[bone_idx]].extend(
                    dround(bone.direction.to_float_array()))

        if not self.map_initialized:
            self.lcoord, self.lcoef = convert_distortion_maps(
                frame.images[0])
            self.rcoord, self.rcoef = convert_distortion_maps(
                frame.images[1])
            self.map_initialized = True
        try:
            image = frame.images[0]
            limage = undistort(image, self.lcoord, self.lcoef, 400, 400)
            image = frame.images[1]
            rimage = undistort(image, self.rcoord, self.rcoef, 400, 400)
        except Exception as e:
            return
        self.pil_limage = limage
        self.pil_rimage = rimage

        self.jd_data = json.dumps(data)


def main():
    try:
        options = parse_args()
        listener = SampleListener()
        controller = Leap.Controller()
        controller.set_policy(Leap.Controller.POLICY_IMAGES)
        controller.set_policy(Leap.Controller.POLICY_BACKGROUND_FRAMES)
        controller.add_listener(listener)
        time.sleep(1)
        host = options.host
        port = options.port
        sock = socket.socket()
        sock.connect((host, port))
        predicts = deque()
        size = 0
        while True:
            data = listener.get_data()
            if not data:
                sock.send('None')
                time.sleep(0.1)
                continue
            data = zlib.compress(pickle.dumps(data))
            sent = sock.send(data)
            if sent == 0:
                raise RuntimeError('socket connection broken')
            chunk = sock.recv(1024)
            if not chunk:
                raise RuntimeError('socket connection broken')
            h = int(chunk) + 1
            predicts.append(h)
            size += 1
            if size > 10:
                predicts.popleft()
                size -= 1
            most = Counter(predicts).most_common(1)
            sys.stdout.write('Current prediction: {:2d}\r'.format(most[0][0]))
            # sys.stdout.write('{:4d} {:30s}\r'.format(most[0][0], str(most)))
    except KeyboardInterrupt:
        pass
    finally:
        controller.remove_listener(listener)


if __name__ == '__main__':
    main()
