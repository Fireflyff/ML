import zmq
import time
context = zmq.Context()

socket=context.socket(zmq.REP)
socket.bind('tcp://127.0.0.1:5559')
while True:
    message = socket.recv()
    print("Received request: ", message)
    time.sleep(1)
    socket.send("World".encode())