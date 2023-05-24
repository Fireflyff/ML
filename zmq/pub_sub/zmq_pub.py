from random import randrange
import zmq

context = zmq.Context()
# 发布人
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:5556")

while True:
    # 订阅号
    zipcode = randrange(1, 100000)
    # 温度
    temperature = randrange(-80, 135)
    # 相对湿度
    relhumidity = randrange(10, 60)

    publisher.send_string(f"{zipcode} {temperature} {relhumidity}")
