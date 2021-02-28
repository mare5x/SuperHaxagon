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
        # print(f"Received {msg}")
        
        # The C++ client sends raw struct bytes as defined in super_client.cpp
        GameState_unpack = "15f"
        msg_type = struct.unpack('=i', msg[:4])[0]
        reply = struct.pack("i", 0)
        if msg_type == 0:  # STATE_ACTION
            _, *state = struct.unpack(f"i{GameState_unpack}", msg)
            action = self.model.get_action(state)
            reply = struct.pack("i", action)
        elif msg_type == 1:  # STATE_EXPERT_ACTION
            _, expert, *state = struct.unpack(f"ii{GameState_unpack}", msg)
            action = self.model.get_action(state, expert)
            reply = struct.pack("i", action)
        elif msg_type == 2:  # EPISODE_SCORE
            _, score = struct.unpack("ii", msg)
            self.model.on_episode_end(score)
        return reply


if __name__ == "__main__":
    context = zmq.Context()
    server = DAGGERServer(context)
    while True:
        server.listen()


