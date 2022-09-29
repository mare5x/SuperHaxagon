from collections import namedtuple, deque
from pathlib import Path 
import random
import math

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter

from plot import plot_queue, moving_average


def plot(ax, data):
    ax.set_title("Q-learning score history")
    x = np.array(data) / 60  # Seconds
    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Attempt number')
    ax.plot(x)
    ax.plot(moving_average(x, k=10))

def state_to_struct(state: list) -> dict:
    # This function must be defined based on the C++ GameState_DQN struct.
    walls = np.array(state[:12]).reshape(6, 2)
    n_slots = state[13:16].index(1) + 4
    cur_slot = next(i for i, x in enumerate(state[16:16+6]) if x > 0)
    player_pos = state[-1] * 6.0
    return {
        "walls": walls,
        "n_slots": n_slots,
        "cur_slot": cur_slot,
        "player_pos": player_pos,
    }

def get_cur_wall_dist(state_struct: dict) -> tuple:
    dist, width = state_struct["walls"][state_struct["cur_slot"]]
    return dist, width

def get_cur_center_offset(state_struct: dict) -> float:
    # -1: left edge, 0: center, 1: right edge
    pos = state_struct["player_pos"] * state_struct["n_slots"] 
    return (pos % 1.0) * 2.0 - 1.0

def plot_state(ax, state):
    # walls array as structured in the C++ code
    walls = np.array(state[:12]).reshape(6, 2)
    ax.set_ylim(0, 1.5)
    ax.bar(list(range(6)), height=walls[:,1], bottom=walls[:,0], width=1.0, align='edge')
    ax.bar(list(range(6)), height=state[16:16+6], bottom=0, width=1.0, align='edge', alpha=0.5)  # Current player slot
    ax.plot(state[-1] * 6.0, 0, 'ro')  # Player position

def latest_checkpoint(path):
    checkpoints = list(path.glob("checkpoint_*.pth"))
    if len(checkpoints) > 0:
        n = max(map(int, (p.stem.rsplit('_')[1] for p in checkpoints)))
        return path / f"checkpoint_{n}.pth"
    if Path(path, "checkpoint.pth").exists():
        return Path(path, "checkpoint.pth")
    return None 


INPUT_SIZE = 6*2 + 1 + 3 + 6 + 1
OUT_SIZE = 3

class SupaNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, OUT_SIZE)
        )

    def forward(self, state):
        return self.net(state)


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory:
    def __init__(self, capacity, strategy="uniform"):
        self.strategy = strategy
        self.memory = deque([], maxlen=capacity)
        self.weights = [1] * capacity

    def push(self, *args):
        item = Transition(*args)
        self.memory.append(item)
        return item

    def sample(self, batch_size):
        if self.strategy == "weighted_negative":
            prev = self.memory[0]
            for i, item in enumerate(self.memory):
                if i == 0 or prev.reward < 0:
                    self.weights[i] = 1
                else:
                    self.weights[i] = self.weights[i - 1] + 1
                prev = item
            if len(self.memory) < self.memory.maxlen:
                weights = self.weights[:len(self.memory)]
            else:
                weights = self.weights
            return random.choices(self.memory, weights=weights, k=batch_size)
        elif self.strategy == "uniform":
            return random.sample(self.memory, batch_size)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def __len__(self):
        return len(self.memory)

    def __repr__(self) -> str:
        return repr(self.memory)

class SupaDQN:
    def __init__(self, experiment_name="exp16"):
        self.experiment_name = experiment_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Double Q-learning uses two networks.
        # target_net is a periodic copy of policy_net.
        self.policy_net = SupaNet().to(self.device)
        self.target_net = SupaNet().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.train()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.criterion = torch.nn.MSELoss()  # TODO: Try huber, mse, l1, ...

        ### Hyperparams
        self.params = {
            "eps_start": 0.99,      # Exploration rate
            "eps_end": 0.0001,
            "eps_decay": 20000,
            "eps_restart": False,    # Restart exploration after this many steps
            "eps_restart_steps": 400_000,
            "target_update": 1000,   # Update target_net to policy_net every this many steps.
            "gamma": 0.95,          # Reward decay rate !!!!!!!!!!!
            "memory_size": 20000,   # Approx. 10 minutes of gameplay
            "memory_sample_strategy": "uniform",  # Memory sampling strategy
            "batch_size": 8,        # Take this many samples from memory for each optimization batch
            "batch_iterations": 2,  # How many training optimization iterations to make for each step
            "reward_slot_center": True,  # Receive reward for being close to the center of a slot
            "reward_slot_center_amount": 0.2,
            "reward_far_wall": True,  # Receive reward for being on a slot where the wall is far away
            "reward_far_wall_amount": 0.5,
            "reward_interval": 60,  # Receive interval_reward reward after this many successful steps
            "interval_reward": 0.5,
            "default_reward": 0.1,  # Receive reward each step
            "loss_reward": -2,      # Reward when game over
        }
        for param, value in self.params.items():
            setattr(self, param, value)
        ###

        self.state = None
        self.action = 0
        self.reward = 0
        self.memory = ReplayMemory(self.memory_size, strategy=self.memory_sample_strategy)
        self.steps_taken = 0
        self.frame_number = 0  # Frame number in episode

        self.is_learning = False 
        self.score_history = []

        # The model works with indices [0, 3), but the server expects [-1,0,1].
        self.actions_tr = [-1, 0, 1]  # Map action index to action
        self.actions_tr_inv = { v: i for i, v in enumerate(self.actions_tr) }

        self.tb_writer = SummaryWriter(f"runs/{self.experiment_name}")  # For tensorboard
        self.checkpoint_enabled = True 
        self.checkpoint_update_interval = 150  # Episodes (deaths)
        self.checkpoint_path = Path(f'./checkpoints/{self.experiment_name}')
        if self.checkpoint_enabled:
            self.load_checkpoint()

    def reset(self):
        self.state = None
        self.action = 0
        self.reward = 0
        self.frame_number = 0

    def exploration_rate(self, t):
        if self.eps_restart:
            t = t % self.eps_restart_steps
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-t / self.eps_decay)

    def pick_action(self, state):
        state = torch.tensor(state).to(self.device)
        if self.is_learning:
            if torch.rand(1) < self.exploration_rate(self.steps_taken):
                return random.randrange(0, 3)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return 

        for it in range(self.batch_iterations):
            batch = self.memory.sample(self.batch_size)
            states = torch.tensor([b.state for b in batch]).to(self.device)
            actions = torch.tensor([b.action for b in batch]).view(-1, 1).to(self.device)
            rewards = torch.tensor([b.reward for b in batch]).view(-1, 1).to(self.device)
            non_final_mask = torch.tensor([b.next_state is not None for b in batch]).to(self.device)
            non_final_next_states = torch.tensor([b.next_state for b in batch if b.next_state is not None]).to(self.device)

            # Pick the Q values of the selected actions for each state
            model_Qs = self.policy_net(states).gather(1, actions)
            # V(s) = 0 if s is final; get best V(s)s
            next_state_Vs = torch.zeros(self.batch_size, 1).to(self.device)
            with torch.no_grad():
                # detach because this happens on the 'target' net, not the online net
                next_state_Vs[non_final_mask] = self.target_net(non_final_next_states).detach().max(1)[0].view(-1, 1)
            target_Qs = (next_state_Vs * self.gamma) + rewards 

            self.optimizer.zero_grad()
            loss = self.criterion(model_Qs, target_Qs)
            loss.backward()
            self.optimizer.step()

            # self.tb_writer.add_scalar("Loss", loss, self.steps_taken)

    def step(self, next_state, reward, done=False):
        self.steps_taken += 1
        if self.steps_taken % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.state is None:  # First frame
            self.state = next_state
            self.action = self.actions_tr_inv[0]
            return self.action

        # Given the reward for the previous action and the new state.
        # When done=True, next_state is None
        # Store (s,a,r,s') into replay memory.
        # N.B. all values are Python types (not tensors).
        self.memory.push(self.state, self.action, reward, next_state)

        self.optimize()

        self.state = next_state 
        if not done:
            self.action = self.pick_action(next_state)
        else:
            self.action = self.actions_tr_inv[0]
        return self.action

    def on_episode_end(self, score=None):
        action = self.actions_tr_inv[0]
        if self.is_learning:
            self.score_history.append(score)
            reward = self.loss_reward
            action = self.step(None, reward, done=True)
            plot_queue.put((plot, self.score_history))

            if self.checkpoint_enabled and len(self.score_history) % self.checkpoint_update_interval == 0:
                self.save_checkpoint()
        
        self.tb_writer.add_scalar("Exploration rate", self.exploration_rate(self.steps_taken), self.steps_taken)
        self.tb_writer.add_scalar("Score", score, self.steps_taken)
        self.tb_writer.flush()
        
        self.reset()
        return self.actions_tr[action]

    def get_action(self, state):
        # plot_queue.put((plot_state, state))
        self.frame_number += 1

        if self.is_learning:
            reward = self.default_reward
            # Reward shaping
            if self.frame_number % self.reward_interval == 0:
                reward += self.interval_reward
            state_struct = state_to_struct(state)
            if self.reward_slot_center:
                center_offset = get_cur_center_offset(state_struct)
                reward += self.reward_slot_center_amount * pow(1 - abs(center_offset), 4)
            if self.reward_far_wall:
                wall_dist, wall_width = get_cur_wall_dist(state_struct)
                if wall_dist > 0.15:
                    reward += self.reward_far_wall_amount * wall_dist
            action = self.step(state, float(reward), done=False)
        else:
            action = self.pick_action(state)
        return self.actions_tr[action]

    def set_is_learning(self, is_learning):
        self.is_learning = is_learning
        self.reset()

    def save_checkpoint(self):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_path / f"checkpoint_{len(self.score_history)}.pth"
        state = {
            'memory': self.memory,
            'score_history': self.score_history,
            'steps_taken': self.steps_taken,
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'params': self.params,
            # scheduler
        }
        torch.save(state, path)
        print(path)

        fig, ax = plt.subplots()
        plot(ax, self.score_history)
        fig.savefig(path.with_suffix(".png"))

        self.tb_writer.add_hparams(self.params, { "hparams/avg_score": sum(self.score_history[-10:]) / 10 }, run_name=f"{self.experiment_name}_{len(self.score_history)}")
        self.tb_writer.flush()

    def load_checkpoint(self, path=None):
        if path is None:
            # Resume latest checkpoint
            path = latest_checkpoint(self.checkpoint_path)
            if path is None:
                return False

        print(path)
        state = torch.load(path)

        self.policy_net.load_state_dict(state['policy_net'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(state['optimizer'])

        self.memory = state['memory']
        self.score_history = state['score_history']
        self.steps_taken = state['steps_taken']

        self.params = state['params']
        for param, value in self.params.items():
            setattr(self, param, value)

        plot_queue.put((plot, self.score_history))
        return True
