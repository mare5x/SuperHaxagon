import time
import struct
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    msg = socket.recv()
    print(f"Received {msg}")
    # print(f"Unpacked: {struct.unpack('i', msg)}")
    msg_type = struct.unpack('=i', msg[:4])[0]
    if msg_type == 0:  # ACTION
        action = struct.unpack(f"2i9d", msg)
        print(f"Got action: {action}")
    else:
        death = struct.unpack("2i", msg)
        print(f"Got death: {death}")

    time.sleep(1)

    socket.send(msg)

