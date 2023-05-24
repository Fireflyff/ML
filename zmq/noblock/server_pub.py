import zmq, time
from random import randrange

context = zmq.Context()

publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:5556")
while True:
    zipcode = 10001
    # 温度
    temperature = randrange(-80, 135)
    # 相对湿度
    relhumidity = randrange(10, 60)

    publisher.send_string(f"{zipcode} {temperature} {relhumidity}")
    print(f"{zipcode} {temperature} {relhumidity}")
    time.sleep(3)