#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-11 14:00
# Last modified: 2017-06-08 19:16
# Filename: recog_twisted.py
# Description:
import optparse, os, json, pickle, zlib

import cv2
import numpy as np

from collections import deque, Counter

from twisted.internet.protocol import ServerFactory, Protocol
from twisted.web import server, resource

from recognition import predict


def parse_args():

    parser = optparse.OptionParser()

    help_msg = "The port to listen on. Default to 9999",
    parser.add_option('--port', type='int', help=help_msg, default=9999)

    help_msg = "The port to serve data on. Default to 8080",
    parser.add_option('--web_port', type='int', help=help_msg, default=8080)

    help_msg = "The interface to listen on. Default is 192.168.2.166."
    parser.add_option('--iface', help=help_msg, default='0.0.0.0')

    help_msg = "The maximum size of the smooth queue."
    parser.add_option('--qsize', help=help_msg, type='int', default=5)

    options, args = parser.parse_args()

    if len(args) != 0:
        parser.error('Invalid Arguments')

    return options


def smooth_position(pos_array):
    m = pos_array.shape[0]
    weights = np.logspace(3, 1, m)
    weights /= weights.sum()
    new_pos = np.average(pos_array, axis=0, weights=weights)
    return new_pos


class RecognitionProtocol(Protocol):
    received_data = b''
    predict_gestures = deque()
    predict_positions = deque()
    size = 0

    def smooth_prediction(self, gesture, pos):
        self.predict_gestures.append(gesture)
        self.predict_positions.append(pos)
        self.size += 1
        while self.size > self.factory.max_size:
            self.predict_gestures.popleft()
            self.predict_positions.popleft()
            self.size -= 1
        pos_array = np.array(self.predict_positions)
        pos = smooth_position(pos_array)
        gesture = Counter(self.predict_gestures).most_common(1)[0][0]
        return (gesture, pos)

    def decode_and_predict(self, decomp_data):
        jds, lbytes, rbytes = pickle.loads(decomp_data, encoding='bytes')
        jd = json.loads(jds.decode('utf-8'))
        limg = np.fromstring(lbytes, dtype=np.uint8).reshape((400, 400))
        rimg = np.fromstring(rbytes, dtype=np.uint8).reshape((400, 400))
        gesture, pos = predict(jd, limg, rimg)
        gesture, pos = self.smooth_prediction(gesture, pos)
        print(gesture, pos)
        return (gesture, pos)

    def dataReceived(self, data):
        if data == b'None':
            self.received_data = b''
            self.predict_gestures.clear()
            self.predict_positions.clear()
            self.size = 0
            for receiver in self.factory.receivers:
                receiver.reset_data()
            return
        try:
            self.received_data += data
            decompressed_data = zlib.decompress(self.received_data)
        except zlib.error:
            return
        self.received_data = b''
        ans, pos = self.decode_and_predict(decompressed_data)
        gesture = str(ans).encode()
        for receiver in self.factory.receivers:
            receiver.handle_data(gesture, pos)

    def connectionMade(self):
        print('Data source connection made')
        self.factory.receivers.append(self)

    def connectionLost(self, reason):
        print('Data source connection lost:', reason)
        self.factory.receivers.remove(self)

    def handle_data(self, data, *more):
        self.transport.write(data)

    def reset_data(self):
        pass


class RecognitionFactory(ServerFactory):
    protocol = RecognitionProtocol 
    receivers = []
    max_size = 10

    def __init__(self, qsize):
        self.max_size = qsize


class WebData(resource.Resource):
    isLeaf = True
    data = ''

    def __init__(self, factory):
        factory.receivers.append(self)

    def handle_data(self, data, *more):
        data = [int(data.decode())]
        data.extend(*more)
        keys = ['gesture', 'xpos', 'ypos', 'height']
        self.data = json.dumps({k: v for k, v in zip(keys, data)})

    def reset_data(self):
        data = {'gesture': -1, 'xpos': 0, 'ypos': 0, 'height': 0}
        self.data = json.dumps(data)

    def render_GET(self, request):
        request.setHeader(b"content-type", b"text/plain")
        return self.data.encode("ascii")


def main():
    options = parse_args()

    from twisted.internet import reactor, endpoints
    factory = RecognitionFactory(options.qsize)
    port = reactor.listenTCP(options.port, factory,
                             interface=options.iface)

    endpoints.serverFromString(reactor, "tcp:"+str(options.web_port)).listen(
       server.Site(WebData(factory)))

    print('Recognition serving on {}, Forward serving on {}.'.format(
        port.getHost(), None))

    reactor.run()


if __name__ == '__main__':
    main()
