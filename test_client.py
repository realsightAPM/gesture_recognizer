#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-11 14:13
# Last modified: 2017-04-11 14:32
# Filename: test_client.py
# Description:
# This is the Twisted Get Poetry Now! client, version 2.1.
import optparse, sys

from random import randint
from time import time

from twisted.internet.protocol import Protocol, ClientFactory


def parse_args():
    parser = optparse.OptionParser()

    _, address = parser.parse_args()

    if not address:
        print(parser.format_help())
        parser.exit()
    address = address[0]

    def parse_address(addr):
        if ':' not in addr:
            host = '127.0.0.1'
            port = addr
        else:
            host, port = addr.split(':', 1)

        if not port.isdigit():
            parser.error('Ports must be integers.')

        return host, int(port)

    return parse_address(address)


class PoetryProtocol(Protocol):

    real = 0
    true_cnt = 0
    cnt = 0
    first_query = 0

    def dataReceived(self, data):
        predict = int(data.decode('utf-8'))
        if predict == self.real:
            self.true_cnt += 1
        self.cnt += 1
        if self.cnt % 1000 == 0:
            now = time()
            elapsed = now - self.first_query
            qps = int(self.cnt / elapsed)
            msg = 'Accuracy: {:.2%}, {:4d} queries per second\r'.format(
                self.true_cnt / self.cnt, qps)
            sys.stdout.flush()
            sys.stdout.write(msg)
        self.query()

    def query(self):
        person = randint(1, 6)
        gesture = randint(1, 10)
        rnd = randint(1, 10)
        fname = 'LP_data/dataset/P{}/G{}/R{}.json'.format(person, gesture, rnd)
        with open(fname) as f:
            data = f.read()
        self.real = gesture - 1
        self.transport.write(data.encode('utf-8'))

    def connectionMade(self):
        self.first_query = time()
        self.query()


class PoetryClientFactory(ClientFactory):

    protocol = PoetryProtocol # tell base class what proto to build

    def clientConnectionFailed(self, connector, reason):
        print('Failed to connect to:', connector.getDestination())
        self.poem_finished()


def poetry_main():
    host, port = parse_args()

    factory = PoetryClientFactory()

    from twisted.internet import reactor

    reactor.connectTCP(host, port, factory)

    reactor.run()


if __name__ == '__main__':
    poetry_main()
