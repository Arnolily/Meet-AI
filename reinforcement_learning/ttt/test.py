import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=300000)
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.target_update_frequency = 2500

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes):
        for e in range(episodes):
            state = env.reset()
            state = np.array(state).flatten()
            done = False
            total_reward = 0

            while not done:
                action = self.act(state)
                next_state, reward, done = env.step(action)
                next_state = np.array(next_state).flatten()
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    print(f"Episode {e+1}/{episodes} - Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")
                    if e % self.target_update_frequency == 0:
                        self.update_target_model()
                    break

            self.replay()

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player 1 is '1', Player 2 is '-1'

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.board

    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            return self.board, -10, True  # Invalid move penalty

        self.board[row, col] = self.current_player

        if self.check_winner(self.current_player):
            return self.board, 10, True  # Win reward

        if np.all(self.board != 0):
            return self.board, 0, True  # Draw

        self.current_player *= -1
        return self.board, 0, False  # No reward yet, game continues

    def check_winner(self, player):
        # Check rows, columns, and diagonals
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if self.board[0, 0] == player and self.board[1, 1] == player and self.board[2, 2] == player:
            return True
        if self.board[0, 2] == player and self.board[1, 1] == player and self.board[2, 0] == player:
            return True
        return False

    def available_actions(self):
        return [i for i in range(9) if self.board[i // 3, i % 3] == 0]

    def render(self):
        print(self.board)

if __name__ == "__main__":
    env = TicTacToe()
    state_size = 9  # 3x3 board flattened
    action_size = 9  # 9 possible moves (0-8)
    agent = DQNAgent(state_size, action_size)
    
    episodes = 100000
    agent.train(env, episodes)
