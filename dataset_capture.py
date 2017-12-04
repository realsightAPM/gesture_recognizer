#!/usr/local/bin/python2
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-06 16:28
# Last modified: 2017-04-11 19:46
# Filename: dataset_capture.py
# Description:
import sys, threading, json, time, os, ctypes

import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib import animation

import Leap


pos_key_idx = {
    1: 'fingers_pro_position',
    2: 'fingers_int_position',
    3: 'fingers_dis_position'}
direction_key_idx = {
    1: 'fingers_pro_direction',
    2: 'fingers_int_direction',
    3: 'fingers_dis_direction'}


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


def filter_image(im):
    a = np.array(im)
    thres = 90
    a[a < thres] = 0
    a[a >= thres] = 255
    return a


class Recorder(object):
    def __init__(self, data_path='LP_data/dataset'):
        self.rnum = 0  # Round num
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        self.data_path = data_path
        ps = [int(k.strip('P')) for k in os.listdir(data_path) if not k.startswith('.')]
        ps.append(0)
        self.pnum = max(ps) + 1

    def new_round(self):
        self.rnum += 1

    def new_person(self):
        self.pnum += 1
        self.rnum = 0

    def set_person(self, pnum):
        self.pnum = pnum
        self.rnum = 0

    def set_round(self, rnum):
        self.rnum = rnum - 1

    def set_gesture(self, gnum):
        self.gnum = gnum
        self.rnum = 0

    def write(self, jd, limage=None, rimage=None):
        if getattr(self, 'gnum', None) is None:
            print 'Please set gesture id first...'
            return False, -1
        self.new_round()
        data_path = self.data_path
        person = 'P{}'.format(self.pnum)
        gesture = 'G{}'.format(self.gnum)
        rnd = 'R{}'.format(self.rnum)
        path = os.path.join(data_path, person, gesture)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, rnd) + '.json', 'w') as f:
            f.write(jd)
        if limage is not None:
            img = Image.fromarray(limage)
            lpath = os.path.join(path, rnd) + '_l.png'
            img.save(lpath)
        if rimage is not None:
            img = Image.fromarray(rimage)
            rpath = os.path.join(path, rnd) + '_r.png'
            img.save(rpath)

        return True, 20 - self.rnum



class SampleListener(Leap.Listener):
    def __init__(self, *args, **kwargs):
        super(SampleListener, self).__init__(*args, **kwargs)
        self.map_initialized = False

    def get_data(self):
        limage = getattr(self, 'pil_limage', None)
        rimage = getattr(self, 'pil_rimage', None)
        jd = getattr(self, 'jd_data', None)
        data = getattr(self, 'data', None)
        return (data, jd, limage, rimage)


    def on_frame(self, controller):
        data = {}
        frame = controller.frame()
        hand = frame.hands[0]
        image = frame.images[0]
        if not frame.is_valid or not image.is_valid or not hand.is_valid:
            return
        data['hand_direction'] = hand.direction.to_float_array()
        data['hand_confidence'] = hand.confidence
        data['hand_sphere_center'] = hand.sphere_center.to_float_array()
        data['hand_sphere_radius'] = hand.sphere_radius

        data['palm_normal'] = hand.palm_normal.to_float_array()
        data['palm_position'] = hand.palm_position.to_float_array()
        data['palm_position_refined'] = hand.stabilized_palm_position.to_float_array()

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
                finger.tip_position.to_float_array())
            data['fingers_tip_direction'].extend(
                finger.direction.to_float_array())
            for bone_idx in range(1, 4):
                bone = finger.bone(bone_idx)
                data[pos_key_idx[bone_idx]].extend(
                    bone.center.to_float_array())
                data[direction_key_idx[bone_idx]].extend(
                    bone.direction.to_float_array())

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
        self.pil_limage = filter_image(limage)
        self.pil_rimage = filter_image(rimage)

        self.jd_data = json.dumps(data)
        self.data = data


def main():
    recorder = Recorder()

    listener = SampleListener()
    controller = Leap.Controller()
    controller.set_policy(Leap.Controller.POLICY_IMAGES)
    controller.add_listener(listener)
    time.sleep(1)

    print 'Default person id would be {}'.format(recorder.pnum)
    print 'Press Enter to quit...'
    print 'Command:'
    print '\tn : Next Person'
    print '\tp[num] : Set person id to [num]'
    print '\tg[num] : Set gesture id to [num]'
    print '\tq : Exit'
    print '\t**null** : Record this frame data'


    try:
        while True:
            cmd = raw_input()
            if cmd == 'n':
                recorder.new_person()
            elif cmd.startswith('p'):
                person = int(cmd[1:])
                recorder.set_person(person)
            elif cmd.startswith('g'):
                gesture = int(cmd[1:])
                recorder.set_gesture(gesture)
            elif cmd == '':
                data, jd, limage, rimage = listener.get_data()
                if not all(k is not None for k in [data, jd, limage, rimage]):
                    continue
                status, _num = recorder.write(jd, limage, rimage)
                if status:
                    print 'Succeed. Remain: {}'.format(_num)
                else:
                    print 'Failed.'
            elif cmd == 'q':
                break
            else:
                print 'Unknown command.'

    except KeyboardInterrupt:
        pass
    finally:
        controller.remove_listener(listener)

if __name__ == '__main__':
    main()
