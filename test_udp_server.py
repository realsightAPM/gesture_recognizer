#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-04-12 18:09
# Last modified: 2017-04-12 20:32
# Filename: test_udp_server.py
# Description:
import socket, random, time


host = '127.0.0.1'
port = 10000
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    s.listen(5)
    print('Waiting for connection')
    while True:
        conn, address = s.accept()
        print('New connection: {}'.format(address))
        try:
            while True:
                conn.send('{}'.format(random.randint(1, 2)).encode('utf-8'))
                time.sleep(0.5)
        except Exception as e:
            print('Lost connection: {}'.format(e))
    s.close()
