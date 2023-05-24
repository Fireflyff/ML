"""
单向数据发布模式-客户端
发布-订阅模式，没有终点也没有起点，就像是一个永无休止的广播
"""
import random
import sys
import time

import zmq

context = zmq.Context()
# 订阅人
subscriber = context.socket(zmq.SUB)
print("尝试连接天气服务器:Collecting updates from weather server...")
subscriber.connect("tcp://localhost:5556")

# 订阅邮政编码
"""sys.argv表示sys模块中的argv变量，
sys.argv是一个字符串的列表，其包含了命令行参数的列表，即使用命令行传递给你的程序的参数。
特别注意：脚本的名称总是sys.argv列表的第一个参数"""
zip_filter = sys.argv[1] if len(sys.argv) > 1 else "10001"

# 使用sub套接字时必须使用zmq_setsockopt设置一个订阅，如果没有设置订阅就不会受到消息
# zip_filter设置一个订阅码，有订阅号是特定值时客户端才可以接收，如果不设置订阅号则设置“”即可
subscriber.setsockopt_string(zmq.SUBSCRIBE, zip_filter)
total_temp = 0
print('连接成功开始接收信息')
for update_nbr in range(5):
    now_string = subscriber.recv_string()
    zipcode, temperature, relhumidity = now_string.split()
    total_temp += int(temperature)
    print(f'接收到{zipcode}')
    print((f"Average temperature for zipcode "
           f"'{zip_filter}' was {total_temp / (update_nbr + 1)} F"))

