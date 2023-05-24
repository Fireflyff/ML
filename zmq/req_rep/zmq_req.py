import zmq
import time
context = zmq.Context()
print("Connecting to hello world server....")
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5559")
for request in range(10):
    print("Sending request ", request, "...")
    socket.send("Hello".encode())
    message = socket.recv()
    print("Received reply ", request,":", message)