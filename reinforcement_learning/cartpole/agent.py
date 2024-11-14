import gym
import numpy as np
import torch
import torch.nn as nn
import random
model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

class agent():
    def __init__(self, model):
        self.qnet = model
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=2e-4)
        self.gamma = 0.99
        self.epsilon = 0.1
        self.qnet = model
    
    # def get_action(self, state):
    #     if np.random.rand() < self.epsilon:
    #         return np.random.randint(2)
    #     else:
    #         state = torch.FloatTensor(state).reshape(1, 4) 
    #         action = self.qnet(state).argmax().item()
    #         return action
        
        
    def train(self, state, action, reward, next_state, done):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move tensors to the appropriate device
        state = state  
        action = action  
        reward = reward  
        next_state = next_state  
        done = done  
        
        # Gather the Q-value for the taken action
        q_value = self.qnet(state).gather(dim=1, index=action)
        
        with torch.no_grad():
            next_q_value = self.qnet(next_state)
        
        # Compute the target Q-value
        next_q_value = next_q_value.max(dim=1)[0].reshape(-1, 1)
        q_target = reward + self.gamma * next_q_value * (1 - done)
        
        # Compute the loss
        loss = torch.nn.MSELoss()(q_value, q_target)
        
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return reward

import pdb
class ReplayBuffer():
    def __init__(self, agent, capacity, env, update):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.env = env
        self.agent = agent
        self.update = update
        
    def load(self):
        old_len = len(self.buffer)
        while len(self.buffer) - old_len < self.update:
            self.buffer.extend(play()[0])
        self.buffer = self.buffer[-self.capacity:]
    
    def __len__(self):
        return len(self.buffer)
    
    def sample(self):
        data = random.sample(self.buffer, 64)
        states = torch.FloatTensor([i[0] for i in data]).reshape(-1, 4)
        actions = torch.LongTensor([i[1] for i in data]).reshape(-1, 1)
        rewards = torch.FloatTensor([i[2] for i in data]).reshape(-1, 1)
        next_states = torch.FloatTensor([i[3] for i in data]).reshape(-1, 4)
        done = torch.LongTensor([i[4] for i in data]).reshape(-1, 1)
        return states, actions, rewards, next_states, done
    
# def test(agent):
#     env = gym.make('CartPole-v1')
#     state = env.reset()[0]
#     done = False
#     total_reward = 0
    
#     while not done:
#         state = torch.FloatTensor(state)
#         action = agent.get_action(state)
#         next_state, reward, done, truncated, info = env.step(action)
#         done = done or truncated
#         total_reward += reward
#         next_state = torch.FloatTensor(next_state)
#         state = next_state
#     return total_reward


def play():
    data = []
    reward_sum = 0

    state, _ = env.reset()
    over = False
    while not over:
        action = model(torch.FloatTensor(state).reshape(1, 4) ).argmax().item()
        if random.random() < 0.1:
            action = env.action_space.sample()

        next_state, reward, over, _, __ = env.step(action)
        data.append((state, action, reward, next_state, over))
        reward_sum += reward

        state = next_state
    return data, reward_sum

class ENV(gym.Wrapper):
    def __init__(self):
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        super().__init__(env)
        self.env = env
        self.step_n = 0
    def reset(self):
        state, _ = self.env.reset()
        self.step_n = 0
        return state, None
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        over = terminated or truncated

        #限制最大步数
        self.step_n += 1
        if self.step_n >= 200:
            over = True
        
        #没坚持到最后,扣分
        if over and self.step_n < 200:
            reward = -1000
        return state, reward, over, truncated, info
    
    
import matplotlib.pyplot as plt
from IPython import display    

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot the scores and mean scores
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.legend()

    # Adjust y-axis limits with padding
    plt.ylim(min(min(scores), min(mean_scores)) - 100, max(max(scores), max(mean_scores)) + 100)
    plt.xlim(0, len(scores) - 1)  # Ensure x-axis limits are correct

    # Add annotations with a slight offset
    if scores:
        plt.text(len(scores) - 1, scores[-1] * 0.95, f"{scores[-1]:.2f}")
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1] * 0.95, f"{mean_scores[-1]:.2f}")

    plt.show(block=False)
    plt.pause(0.1)

    
    
from tqdm import tqdm
if __name__ == '__main__':
    

    env = ENV()
    agent = agent(model)
    buffer = ReplayBuffer(agent, env=env, update = 200, capacity = 20000)
    score = []
    
    for _ in range(10):
        
        for i in tqdm(range(100)):
            buffer.load()
            total_reward = 0
            
            for e in range(200):
                reward = 0
                states, actions, rewards, next_state, done = buffer.sample()
                reward = agent.train(states, actions, rewards, next_state, done)
                total_reward += reward.mean().item()

            score.append(total_reward)
            mean_scores = [sum(score[:i+1]) / (i+1) for i in range(len(score))]
            plot(score, mean_scores)
        test_result = sum([play()[-1] for _ in range(20)]) / 20
        print(i, len(buffer), test_result)
