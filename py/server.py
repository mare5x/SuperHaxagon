""" Connection between the DLL process and Python using ZeroMQ.

    Reply to SuperHaxagon requests, receive current state information
    and reply with an action (left/right/none).
    Why? Easier model fitting using Python than C++.
"""


import struct

import zmq

from learn import DAGGER


class ReplyServer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

    def listen(self):
        msg = self.socket.recv()
        msg = self.process_msg(msg)
        self.socket.send(msg)

    def process_msg(self, msg):
        return msg


class DAGGERServer(ReplyServer):
    def __init__(self, ctx):
        super().__init__(ctx)
        
        self.model = DAGGER.load()

    def process_msg(self, msg):
        print(f"Received {msg}")
        
        # msg_type = struct.unpack('=i', msg[:4])[0]
        # if msg_type == 0:  # ACTION
        #     action = struct.unpack(f"2i15f", msg)
        #     print(f"Got action: {action}")
        # elif msg_type == 1:
        #     death = struct.unpack("2i", msg)
        #     print(f"Got death: {death}")
        # return msg

        return struct.pack("i", 0)


if __name__ == "__main__":
    context = zmq.Context()
    server = DAGGERServer(context)
    while True:
        server.listen()
