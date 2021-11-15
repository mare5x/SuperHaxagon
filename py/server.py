""" Connection between the DLL process and Python using ZeroMQ.

    Reply to SuperHaxagon requests, receive current state information
    and reply with an action (left/right/none).
    Why? Easier model fitting using Python than C++.
"""

import struct

import zmq

import learn


MSG_CODES = {
    0: "ALL",
    1: "DAGGER",
    2: "DQN",

    3: "DAGGER_STATE_ACTION",
    4: "DAGGER_STATE_EXPERT_ACTION",
    5: "EPISODE_SCORE",
    6: "DQN_STATE_ACTION",
}


class SuperServer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

        self.dagger = DAGGERServer()

        self.current_server = None

    def listen(self):
        msg = self.socket.recv()
        reply = struct.pack("i", 0)

        msg_type = MSG_CODES[struct.unpack('=i', msg[:4])[0]]
        if msg_type.startswith("DAGGER"):
            self.current_server = self.dagger 
        elif msg_type.startswith("DQN"):
            pass # self.current_server = self.dqn

        if self.current_server is not None:
            reply = self.current_server.process_msg(msg)
        
        self.socket.send(reply)


class DAGGERServer:
    def __init__(self):
        self.model = learn.DAGGER.load()
        self.start_plotting()

    def process_msg(self, msg):
        # print(f"Received {msg}")
        
        # The C++ client sends raw struct bytes as defined in super_client.cpp
        GameState_unpack = "15f"
        msg_type = MSG_CODES[struct.unpack('=i', msg[:4])[0]]
        reply = struct.pack("i", 0)
        if msg_type == "DAGGER_STATE_ACTION":
            _, *state = struct.unpack(f"i{GameState_unpack}", msg)
            action = self.model.get_action(state)
            reply = struct.pack("i", action)
        elif msg_type == "DAGGER_STATE_EXPERT_ACTION":
            _, expert, *state = struct.unpack(f"ii{GameState_unpack}", msg)
            action = self.model.get_action(state, expert)
            reply = struct.pack("i", action)
        elif msg_type == "EPISODE_SCORE":
            _, score = struct.unpack("ii", msg)
            self.model.on_episode_end(score)
        return reply

    def start_plotting(self):
        learn.plot_queue.put(self.model.score_history)
        learn.start_plotting()


class DQNServer:
    def __init__(self):
        pass 

    def process_msg(self, msg):
        GameState_unpack = f"{6*2 + 1 + 3 + 6 + 1}f"
        msg_type = MSG_CODES[struct.unpack('=i', msg[:4])[0]]
        reply = struct.pack("i", 0)
        if msg_type == "DQN_STATE_ACTION":
            _, *state = struct.unpack(f"i{GameState_unpack}", msg)
            action = self.model.get_action(state)
            reply = struct.pack("i", action)
        elif msg_type == "EPISODE_SCORE":
            _, score = struct.unpack("ii", msg)
            self.model.on_episode_end(score)
        return reply


if __name__ == "__main__":
    context = zmq.Context()
    server = SuperServer(context)
    while True:
        server.listen()
