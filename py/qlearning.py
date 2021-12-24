from collections import namedtuple, deque
from pathlib import Path 
import random
import math

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 

from plot import plot_queue, moving_average


def plot(ax, data):
    ax.set_title("Q-learning score history")
    x = np.array(data) / 60  # Seconds
    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Attempt number')
    ax.plot(x)
    ax.plot(moving_average(x, k=10))

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
            nn.Linear(INPUT_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, OUT_SIZE)
        )

    def forward(self, state):
        return self.net(state)


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        item = Transition(*args)
        self.memory.append(item)
        return item

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __repr__(self) -> str:
        return repr(self.memory)

class SupaDQN:
    def __init__(self):
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

        self.state = None
        self.action = 0
        self.reward = 0

        self.steps_taken = 0
        self.eps_start = 0.9  # Exploration rate
        self.eps_end = 0.05
        self.eps_decay = 10000
        self.target_update = 2046  # Update target_net to policy_net every this many steps.

        self.gamma = 0.95

        self.memory = ReplayMemory(20000)  # Approx. 10 minutes of gameplay
        self.batch_size = 1024

        self.is_learning = False 
        self.score_history = []

        # The model works with indices [0, 3), but the server expects [-1,0,1].
        self.actions_tr = [-1, 0, 1]  # Map action index to action
        self.actions_tr_inv = { v: i for i, v in enumerate(self.actions_tr) }

        self.checkpoint_enabled = True 
        self.checkpoint_update_interval = 150  # Episodes (deaths)
        self.checkpoint_path = Path('./checkpoints/')
        if self.checkpoint_enabled:
            self.load_checkpoint()

    def reset(self):
        self.state = None
        self.action = 0
        self.reward = 0

    def exploration_rate(self, t):
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

        loss = self.criterion(model_Qs, target_Qs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
            return self.action
        else:
            # print(self.state, self.action, reward, next_state)
            print(self.exploration_rate(self.steps_taken))
            return self.actions_tr_inv[0]

    def on_episode_end(self, score=None):
        action = self.actions_tr_inv[0]
        if self.is_learning:
            self.score_history.append(score)
            reward = -1.0
            action = self.step(None, reward, done=True)
            plot_queue.put((plot, self.score_history))

            if self.checkpoint_enabled and len(self.score_history) % self.checkpoint_update_interval == 0:
                self.save_checkpoint()
        self.reset()
        return self.actions_tr[action]

    def get_action(self, state):
        # plot_queue.put((plot_state, state))

        if self.is_learning:
            reward = 0
            action = self.step(state, reward, done=False)
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
            'optimizer': self.optimizer.state_dict()
            # scheduler
        }
        torch.save(state, path)
        print(path)

        fig, ax = plt.subplots()
        plot(ax, self.score_history)
        fig.savefig(path.with_suffix(".png"))

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

        plot_queue.put((plot, self.score_history))
        return True
