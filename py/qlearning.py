from collections import namedtuple, deque
import random
import math

import torch
import torch.nn as nn


INPUT_SIZE = 6*2 + 1 + 3 + 6 + 1
OUT_SIZE = 3

class SupaNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, OUT_SIZE)
        )

    def forward(self, state):
        return self.net(state)


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __repr__(self) -> str:
        return repr(self.memory)

class SupaDQN:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = SupaNet()
        self.policy_net.to(self.device)
        self.policy_net.train()

        self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.0003)
        self.criterion = torch.nn.SmoothL1Loss()  # TODO: Try huber, mse, l1, ...

        self.state = None
        self.action = 0
        self.reward = 0

        self.steps_taken = 0
        self.eps_start = 0.9  # Exploration rate
        self.eps_end = 0.05
        self.eps_decay = 10000

        self.gamma = 0.999

        self.memory = ReplayMemory(10000)
        self.batch_size = 128

        self.is_learning = False 

        # The model works with indices [0, 3), but the server expects [-1,0,1].
        self.actions_tr = [-1, 0, 1]  # Map action index to action
        self.actions_tr_inv = { v: i for i, v in enumerate(self.actions_tr) }

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
            next_state_Vs[non_final_mask] = self.policy_net(non_final_next_states).detach().max(1)[0].view(-1, 1)
        target_Qs = (next_state_Vs * self.gamma) + rewards 

        loss = self.criterion(model_Qs, target_Qs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self, next_state, reward, done=False):
        self.steps_taken += 1

        if self.state is None:  # First frame
            self.state = next_state
            self.action = self.actions_tr_inv[0]
            return self.action

        # Given the reward for the previous action and the new state.
        # When done=True, next_state is None
        # Store (s,a,r,s') into replay memory.
        # N.B. all values are Python types (not tensors).
        if reward < 0: 
            print(self.state, self.action, reward, next_state)
            print(self.exploration_rate(self.steps_taken))
        self.memory.push(self.state, self.action, reward, next_state)

        self.optimize()

        self.state = next_state 
        if not done:
            self.action = self.pick_action(next_state)
            return self.action
        else:
            return self.actions_tr_inv[0]

    def on_episode_end(self, score=None):
        action = self.actions_tr_inv[0]
        if self.is_learning:
            reward = -1.0
            action = self.step(None, reward, done=True)
        self.reset()
        return self.actions_tr[action]

    def get_action(self, state):
        if self.is_learning:
            reward = 0
            action = self.step(state, reward, done=False)
        else:
            action = self.pick_action(state)
        return self.actions_tr[action]

    def set_is_learning(self, is_learning):
        self.is_learning = is_learning
        self.reset()
