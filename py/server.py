""" Connection between the DLL process and Python using ZeroMQ.

    Reply to SuperHaxagon requests, receive current state information
    and reply with an action (left/right/none).
    Why? Easier model fitting using Python than C++.
"""
import struct

import dagger
import qlearning 
import sb3_rl
import plot

import zmq


MSG_CODES = {
    0: "ALL",
    1: "DAGGER",
    2: "DQN",

    3: "DAGGER_STATE_ACTION",
    4: "DAGGER_STATE_EXPERT_ACTION",
    5: "EPISODE_SCORE",
    6: "DQN_STATE_ACTION",
    7: "DQN_LEARNING_MODE"
}

class SuperServer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

        self.dagger = DAGGERServer()
        self.dqn = DQNServer()
        # self.dqn = SB3OptunaServer()

        self.current_server = None

    def listen(self):
        msg = self.socket.recv()
        reply = struct.pack("i", 0)

        msg_type = MSG_CODES[struct.unpack('=i', msg[:4])[0]]
        if msg_type.startswith("DAGGER"):
            self.current_server = self.dagger 
        elif msg_type.startswith("DQN"):
            self.current_server = self.dqn

        if self.current_server is not None:
            reply = self.current_server.process_msg(msg)
        
        self.socket.send(reply)


class DAGGERServer:
    def __init__(self):
        # pass 
        self.model = dagger.DAGGER.load()
        # plot.plot_queue.put((dagger.plot, self.model.score_history))

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


class DQNServer:
    def __init__(self):
        # self.model = qlearning.SupaDQN()
        self.model = sb3_rl.SupaSB3()

    def process_msg(self, msg):
        GameState_unpack = f"{qlearning.INPUT_SIZE}f"
        msg_type = MSG_CODES[struct.unpack('=i', msg[:4])[0]]
        reply = struct.pack("i", 0)
        if msg_type == "DQN_STATE_ACTION":
            _, *state = struct.unpack(f"i{GameState_unpack}", msg)
            action = self.model.get_action(state)
            reply = struct.pack("i", action)
        elif msg_type == "EPISODE_SCORE":
            _, score = struct.unpack("ii", msg)
            self.model.on_episode_end(score)
        elif msg_type == "DQN_LEARNING_MODE":
            _, mode = struct.unpack("ii", msg)
            self.model.set_is_learning(mode == 1)
        return reply


class SB3OptunaServer:
    def __init__(self):
        self.model = sb3_rl.SupaSB3Optuna()

    def process_msg(self, msg):
        GameState_unpack = f"{qlearning.INPUT_SIZE}f"
        msg_type = MSG_CODES[struct.unpack('=i', msg[:4])[0]]
        reply = struct.pack("i", 0)
        if msg_type == "DQN_STATE_ACTION":
            _, *state = struct.unpack(f"i{GameState_unpack}", msg)
            action = self.model.get_action(state)
            reply = struct.pack("i", action)
        elif msg_type == "EPISODE_SCORE":
            _, score = struct.unpack("ii", msg)
            self.model.on_episode_end(score)
        elif msg_type == "DQN_LEARNING_MODE":
            _, mode = struct.unpack("ii", msg)
            self.model.set_is_learning(mode == 1)
        return reply


if __name__ == "__main__":
    context = zmq.Context()
    server = SuperServer(context)
    # plot.start_plotting()
    while True:
        server.listen()
