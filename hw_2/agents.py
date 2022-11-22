import random
import numpy as np

from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, lr_scheduler

class RandomAgent:
    def __init__(self):
        self.is_eval = True
        
    def choose_action(self, env):
        obs, actions = env.getHash(), env.getEmptySpaces()
        actions = [env.int_from_action(action) for action in actions]
        action = np.random.choice(actions)
        tup_action = env.action_from_int(action)
        return tup_action

class TabQAgent:
    def __init__(self, q_size, alpha, gamma, 
                 eps=0.05):
        self.q_size = q_size
        self.q = defaultdict(self.init_q)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.is_eval = False
        self.reset()
        
    def reset(self):
        self.hist = []

    def init_q(self):
        return np.zeros(self.q_size)         
    
    def choose_action(self, env):
        state, actions = env.getHash(), env.getEmptySpaces()
        actions = [env.int_from_action(action) for action in actions]

        q_vals = self.q[state][np.array(actions)]
        idx = np.argmax(q_vals)
        action = actions[idx]
        if not self.is_eval and np.random.uniform() < self.eps:
            action = np.random.choice(actions)
        self.hist.append([state, action])
        tup_action = env.action_from_int(action)
        return tup_action
    
    def update_q(self, reward, state, action, next_state):
        if state is not None:
            self.q[state][action] = self.q[state][action] + self.alpha * (
                reward \
                + self.gamma * np.max(self.q[next_state]) - self.q[state][action]
            )
    
    def learn(self, reward):
        self.hist.reverse()
        next_state = None
        for i, (state, action) in enumerate(self.hist):
            if i == 0:
                self.q[state][action] = reward
            else:
                self.update_q(0, state, action, next_state)
            next_state = state

class ExpirienceReplay(deque):
    def sample(self, size):
        batch = random.choices(self, k=size)
        return list(zip(*batch))

class DQNAgent:
    def __init__(self, model, hid_size, game_size, action_size, buffer_size, batch_size, gamma, lr, eps=0.05):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.model = model(hid_size, game_size, action_size)
        self.target_model = model(hid_size, game_size, action_size)
        self.action_size = action_size
        self.eps = eps
        self.is_eval = False
        self.buffer = ExpirienceReplay(maxlen=self.buffer_size)
        self.hist = []

        self.model.to(self.device)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.8)
    
    def consume(self, reward):
        if len(self.hist):
            self.hist.reverse()
            next_state = None
            prev_actions = None
            for i, (state, action, poss_actions) in enumerate(self.hist):
                if i == 0:
                    self.buffer.append((state, action, state, reward, 1, poss_actions))
                else:
                    self.buffer.append((state, action, next_state, 0, 0, poss_actions))
                next_state = state
                prev_actions = poss_actions
        self.hist = []
 
    def sample_batch(self):
        batch = self.buffer.sample(self.batch_size) 
        state, action, next_state, reward, done, poss_actions = batch
        state = torch.tensor(np.array(state, dtype=np.float32)).to(self.device)
        action = torch.tensor(np.array(action, dtype=np.int64)).to(self.device)
        next_state = torch.tensor(np.array(next_state, dtype=np.float32)).to(self.device)
        reward = torch.tensor(np.array(reward, dtype=np.float32)).to(self.device)
        done = torch.tensor(np.array(done, dtype=np.float32)).to(self.device)
        poss_actions = torch.tensor(np.array(poss_actions, dtype=np.bool_)).to(self.device)
        return (state, action, next_state, reward, done, poss_actions)
    
    def choose_action(self, env):
        obs, actions = env.getState2d(), env.getEmptySpaces()
        actions = [env.int_from_action(action) for action in actions]
        self.model.eval()
        actions = np.array(actions)
        state = torch.tensor(np.array(obs, dtype=np.float32)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.model(state)
        q_vals = q_vals.detach().cpu().numpy().reshape(-1)
        q_vals = q_vals[actions]
        action = actions[np.argmax(q_vals)]
        if not self.is_eval and np.random.uniform() < self.eps:
            action = np.random.choice(actions)
        if not self.is_eval:
            poss_actions = np.zeros(self.action_size)
            poss_actions[actions] = 1
            self.hist.append([obs, action, poss_actions])
        tup_action = env.action_from_int(action)
        return tup_action

    def train_step(self):
        (state, action, next_state, reward, done, actions) = self.sample_batch()
        with torch.no_grad():
            next_actions = self.model(next_state).detach()
            next_actions[actions] = -10 ** 6
            next_action = next_actions.argmax(dim=1)
            q_targets_next = self.target_model(next_state).gather(1, next_action.view(-1, 1))

        self.model.train()
        q_targets = reward.view(-1, 1) + self.gamma * q_targets_next * (1 - done.view(-1, 1))
        batch_action = action.view(-1, 1).type(torch.int64)
        q_expected = self.model(state).gather(1, batch_action)
        loss = self.criterion(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()   