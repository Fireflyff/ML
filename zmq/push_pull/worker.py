""""
并行任务工人
将PULL套接字连接到tcp://localhost:5557
通过上面的套接字来手机自来发生器的工作负载
将PUSH套接字连接到tcp://localhost:5558
通过5558的套接字发送结果给接收器
"""
import sys
import time

import zmq

context = zmq.Context()

# 用于接收消息的套接字
receiver = context.socket(zmq.PULL)
receiver.connect('tcp://localhost:5557')

# 用于发送消息的套接字
sender = context.socket(zmq.PUSH)
sender.connect('tcp://localhost:5558')

# 永远地处理任务
while True:
    print('-----------------------------')
    print('接收信息')
    s = receiver.recv()
    # 用于查看器的简易过程指示器
    sys.stdout.write('.')
    sys.stdout.flush()
    # 不做工作
    time.sleep(int(s) * 0.0000001)
    # 将结果发送给接收器
    sender.send_string(f'{s}')
    print('发送消息')

