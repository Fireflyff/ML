"""
并行任务接收器
将PULL套接字绑定到 tcp://localhost:5558
通过上述套接字收集来自各个工人的结果
"""
import sys
import time

import zmq

# 准备上下文和套接字
context = zmq.Context()
receiver = context.socket(zmq.PULL)
# 接收各个工人结果的服务器
receiver.bind('tcp://*:5558')
# 等待批次的开始

s = receiver.recv()

# 启动时钟
tstart_time = time.time()

# 处理100个确认
for task_nbr in range(1):
    print('-----------------------------')
    print('开始接收任务')
    s = receiver.recv()
    print(s)
    if task_nbr % 10 == 0:
        sys.stdout.write(':')
    else:
        sys.stdout.write('.')
    sys.stdout.flush()
    print('准备接收下一个任务')

# 计算并报告批次的用时
tend=time.time()
print(f"Total elapsed time: {(tend-tstart_time)*1000} msec")
