# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:27:30 2023

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
import seaborn as sns

class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()

        self.size = 12
        self.start = (0, 0)
        self.state = self.start
        self.goal_positions = []  
        self.obstacle_positions = []

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype='int64')

        self.generate_maze()

    def generate_maze(self):
        np.random.seed(2023)
        free_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        free_positions.remove(self.start)

        num_obstacles = self.size * self.size // 16
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
         
         
class DQNAgent:
    def __init__(self, env, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.8
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 5e-5
        self.memory = deque(maxlen=2000)
        self.batch_size = 128
        self.max_steps = env.size * env.size * 4
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
        
        
        
# Training function for the DQN agent

def train_dqn(env, agent, episodes, target_update_interval=40):
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            
            step_count += 1
            if step_count >= agent.max_steps:
                break
            
        rewards.append(total_reward)

        if (episode + 1) % target_update_interval == 0:
            agent.update_target_network()
            print(f"Episode: {episode + 1}, Average Reward: {np.mean(rewards[-100:])}")
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    return rewards, agent.q_network

def extract_policy(q_network, env, agent):
    policy = np.zeros((env.size, env.size), dtype=int)
    q_table = np.zeros((env.size, env.size, env.action_space.n))

    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = q_network(state_tensor)
            q_values_np = q_values.detach().cpu().numpy()

            q_table[state] = q_values_np
            policy[state] = np.argmax(q_values_np)

    return policy, q_table

# Plot the training curve
def plot_training_curve(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Curve')
    plt.show()
 
def visualize_q_table(q_table):
    fig, axes = plt.subplots(nrows=q_table.shape[2], ncols=1, figsize=(10, 6 * q_table.shape[2]))
    for i, ax in enumerate(axes):
        sns.heatmap(q_table[:, :, i], annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title(f"Action {i}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    plt.tight_layout()
    plt.show()
        
def visualize_policy(env, policy):
    arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    policy_arrows = np.vectorize(arrow_map.get)(policy)

    env_map = np.zeros_like(policy, dtype=object)

    for i in range(env.size):
        for j in range(env.size):
            position = (i, j)
            if position == env.start:
                env_map[env.size - 1 - i, j] = 'S'
            elif position in env.goal_positions:
                env_map[env.size - 1 - i, j] = 'G'
            elif position in env.obstacle_positions:
                env_map[env.size - 1 - i, j] = 'X'
            else:
                env_map[env.size - 1 - i, j] = policy_arrows[i, j]

    plt.figure(figsize=(10, 6))
    sns.heatmap(np.zeros_like(env_map, dtype=float), annot=env_map, fmt='', cmap="coolwarm", cbar=False, xticklabels=True, yticklabels=True, linewidths=0.5, linecolor='black')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Policy with Environment")
    plt.gca().invert_yaxis()
    plt.show()

    
# Create the maze environment
maze_env = MazeEnv()

# Create the DQN agent
state_size = maze_env.observation_space.shape[0]
action_size = maze_env.action_space.n
dqn_agent = DQNAgent(maze_env, state_size, action_size)

# Train the agent
num_episodes = 1000
rewards, trained_q_network = train_dqn(maze_env, dqn_agent, num_episodes )
policy, q_table = extract_policy(trained_q_network, maze_env,dqn_agent )

##visualize performance
plot_training_curve(rewards)
##visualize the q table
visualize_q_table(q_table)
##visualize the policy
visualize_policy(maze_env,policy)
