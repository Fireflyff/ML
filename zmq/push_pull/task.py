# 生成并行任务端
import random
import time

import zmq

context = zmq.Context()
# 用于发送信息的套接字
sender = context.socket(zmq.PUSH)
sender.bind("tcp://*:5557")

# 用于发送批次开始消息的套接字
sink = context.socket(zmq.PUSH)
sink.connect("tcp://localhost:5558")
print("当工人准备好后点击回程Press Enter when the workers are ready: ")
_ = input()
print("开始发送任务Sending tasks to workers...")
# 第一个消息是“0”，他表示批次的开始
sink.send(b'0')
# 初始化随机数发生器
random.seed()
total_msec = 0
print('准备进入循环')
# 发送一百个任务
for task_nbr in range(1):
    print('-----------------------------')
    print(f'开始发送{task_nbr}')
    # 从1到100毫秒的随机工作负载
    workload = random.randint(1, 100)
    total_msec += workload
    sender.send_string(f"{workload}")

print(f"总体耗费时间为Total expected cost: {total_msec} msec")
# 给ZMQ一点时间来传递
time.sleep(1)
