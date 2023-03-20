# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 20:56:38 2023

@author: 51027
"""

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class MazeEnv(gym.Env):
    def __init__(self):
        self.size = 14
        self.start = (0, 0)
        self.goal_positions = []
        self.obstacle_positions = []
        self.state = self.start
        self.generate_maze()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.size),
            spaces.Discrete(self.size),
        ))

    def generate_maze(self):
        np.random.seed(2023)
        free_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        free_positions.remove(self.start)

        num_obstacles = self.size * self.size // 8
        self.obstacle_positions = [free_positions.pop(np.random.randint(len(free_positions))) for _ in range(num_obstacles)]

        for _ in range(2):
            goal = free_positions.pop(np.random.randint(len(free_positions)))
            self.goal_positions.append(goal)

    def step(self, action):
        x, y = self.state

        if action == 0:  # up
            x = max(x - 1, 0)
        elif action == 1:  # down
            x = min(x + 1, self.size - 1)
        elif action == 2:  # left
            y = max(y - 1, 0)
        elif action == 3:  # right
            y = min(y + 1, self.size - 1)

        new_state = (x, y)
        if new_state in self.obstacle_positions:
            new_state = self.state

        reward = 0
        done = False
        if new_state in self.goal_positions:
            reward = 1
            done = True

        self.state = new_state
        return new_state, reward, done, {}

    def reset(self):
        self.state = self.start
        return self.state

    def render(self, mode='human'):
         fig, ax = plt.subplots(figsize=(8, 8))
         for i in range(self.size):
             for j in range(self.size):
                 if (i, j) in self.obstacle_positions:
                     obstacle = patches.Rectangle((j, self.size - 1 - i), 1, 1, facecolor='black')
                     ax.add_patch(obstacle)
                 elif (i, j) in self.goal_positions:
                     goal = patches.Rectangle((j, self.size - 1 - i), 1, 1, facecolor='green')
                     ax.add_patch(goal)
                 elif (i, j) == self.state:
                     agent = patches.Rectangle((j, self.size - 1 - i), 1, 1, facecolor='blue')
                     ax.add_patch(agent)
         plt.xticks(range(self.size), fontsize=8)
         plt.yticks(range(self.size), fontsize=8)
         plt.xlim(0, self.size)
         plt.ylim(0, self.size)
         plt.grid()
         plt.gca().invert_yaxis()
         plt.show()
        

def choose_action(state, q_table, epsilon, n_actions):
    if np.random.random() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(q_table[state])

def q_learning(env, episodes, alpha, gamma, epsilon, progress_interval=50):
    n_actions = env.action_space.n
    q_table = np.zeros((env.size, env.size, n_actions))

    rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = choose_action(state, q_table, epsilon, n_actions)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Update Q-table
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            state = next_state

        rewards.append(total_reward)
        
        if (episode + 1) % progress_interval == 0:
            avg_reward = np.mean(rewards[-progress_interval:])
            print(f"Episode: {episode + 1}, Average Reward: {avg_reward}")

    return {'q_table': q_table, 'rewards': rewards}

def sarsa(env, episodes, alpha, gamma, epsilon):
    n_actions = env.action_space.n
    q_table = np.zeros((env.size, env.size, n_actions))

    rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        action = choose_action(state, q_table, epsilon, n_actions)
        while not done:
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            next_action = choose_action(next_state, q_table, epsilon, n_actions)
            # Update Q-table
            q_table[state][action] += alpha * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])

            state = next_state
            action = next_action

        rewards.append(total_reward)

    return {'q_table': q_table, 'rewards': rewards}

def expected_sarsa(env, episodes, alpha, gamma, epsilon):
    n_actions = env.action_space.n
    q_table = np.zeros((env.size, env.size, n_actions))

    rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = choose_action(state, q_table, epsilon, n_actions)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Calculate expected Q-value for the next state
            expected_q = np.sum(q_table[next_state] * ((epsilon / n_actions) * np.ones(n_actions) + (1 - epsilon) * np.eye(n_actions)[np.argmax(q_table[next_state])]))

            # Update Q-table
            q_table[state][action] += alpha * (reward + gamma * expected_q - q_table[state][action])
            state = next_state

        rewards.append(total_reward)

    return {'q_table': q_table, 'rewards': rewards}

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=2000)
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = self.build_model().to(self.device)
        self.target_network = self.build_model().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.detach().cpu().numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = next_q_values.max(1)[0]
        target = rewards + (1 - dones.float()) * self.gamma * next_q_values

        loss = nn.functional.smooth_l1_loss(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)

    def load(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath))




def plot_q_table(q_table):
    q_table_vis = np.max(q_table, axis=2)
    plt.figure(figsize=(8, 8))
    plt.imshow(q_table_vis, cmap='viridis')
    plt.colorbar()
    plt.title("Q-table (max Q-value for each state)")
    plt.show()

def plot_training_curve(rewards, title='Training Curve'):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.show()
    

maze_env = MazeEnv()

q_learning_agent = q_learning(maze_env, episodes=10000, alpha=0.3, gamma=0.99, epsilon=0.5)
sarsa_agent = sarsa(maze_env, episodes=100, alpha=0.1, gamma=0.99, epsilon=0.1)
expected_sarsa_agent = expected_sarsa(maze_env, episodes=100, alpha=0.1, gamma=0.99, epsilon=0.1)

plot_q_table(q_learning_agent['q_table'])
plot_training_curve(q_learning_agent['rewards'])

plot_q_table(sarsa_agent['q_table'])
plot_training_curve(sarsa_agent['rewards'])

plot_q_table(expected_sarsa_agent['q_table'])
plot_training_curve(expected_sarsa_agent['rewards'])