import zmq, time
from random import randrange

context = zmq.Context()

sink = context.socket(zmq.REP)
sink.bind("tcp://127.0.0.1:5558")
message=sink.recv()
print(message)
# time.sleep(5)
sink.send("World".encode())