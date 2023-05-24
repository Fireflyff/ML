#coding=utf-8
'''
Created on 2015-10-13
在这里，同时处理多个套接字，那么接收消息的时候，就需要设置noblock
不然会在第一个接收消息的地方堵塞
@author: kwsy2015
'''
import zmq
import time
 
# Prepare our context and sockets
context = zmq.Context()
 
# Connect to task ventilator
receiver = context.socket(zmq.REQ)
receiver.connect("tcp://127.0.0.1:5558")
receiver.send('Hello'.encode())
# time.sleep(5)
# Connect to weather server
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://localhost:5556")
subscriber.setsockopt_string(zmq.SUBSCRIBE, "10001")
#
# msg1 = receiver.recv()
#
# msg2 = subscriber.recv_string()


# msg1 = receiver.recv(zmq.NOBLOCK)
# msg2 = subscriber.recv(zmq.NOBLOCK)
# print(msg1, msg2)

# Process messages from both sockets
# We prioritize traffic from the task ventilator
#################################################################
########################### NOBLOCK #############################
'''
while True:
 
    # Process any waiting tasks
    while True:
        try:
            #用了NOBLOCK，就意味着得不到消息时不会堵塞在这里
            msg = receiver.recv(zmq.NOBLOCK)
            print(msg)
        except zmq.ZMQError:
            break
        # process task
 
    # Process any waiting weather updates
    while True:
        try:
            msg = subscriber.recv_string(zmq.NOBLOCK)
            print(msg)
        except zmq.ZMQError:
            break
        # process weather update
 
    # No activity, so sleep for 1 msec
    time.sleep(0.001)
'''
####################################################################
# Initialize poll set
poller = zmq.Poller()
poller.register(receiver, zmq.POLLIN)
poller.register(subscriber, zmq.POLLIN)
 
# Process messages from both sockets
while True:
    try:
        socks = dict(poller.poll())
    except KeyboardInterrupt:
        break
    print(socks)
    if receiver in socks:
        message = receiver.recv()
        # process task
 
    if subscriber in socks:
        message = subscriber.recv()
        # process weather update