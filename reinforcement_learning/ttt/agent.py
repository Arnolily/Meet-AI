import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import torch.nn.functional as F
import collections
BOARD_ROWS = 3
BOARD_COLS = 3
MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.0001
EPSILON_START = 1.0  # Exploration rate
EPSILON_END = 0.1
EPSILON_DECAY = 0.995


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        self.isEnd = False
        self.playerSymbol = 1  # Player 1 starts
    
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        self.isEnd = False
        self.playerSymbol = 1
        return self.board.flatten()
    
    def availablePositions(self):
        return [(i, j) for i in range(BOARD_ROWS) for j in range(BOARD_COLS) if self.board[i, j] == 0]

    def step(self, action):
        x = action // BOARD_COLS
        y = action % BOARD_COLS
        self.board[x, y] = self.playerSymbol
        reward, done = self.check_winner()
        self.playerSymbol = -self.playerSymbol  # Switch player
        return self.board.flatten(), reward, done

    def check_winner(self):
        for i in range(BOARD_ROWS):
            if abs(sum(self.board[i, :])) == 3:
                return 1, True
        for j in range(BOARD_COLS):
            if abs(sum(self.board[:, j])) == 3:
                return 1, True
        if abs(sum(self.board[i, i] for i in range(BOARD_ROWS))) == 3 or \
           abs(sum(self.board[i, BOARD_ROWS - i - 1] for i in range(BOARD_ROWS))) == 3:
            return 1, True
        if len(self.availablePositions()) == 0:
            return 0.5, True  # Draw
        return 0, False  # Continue

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(BOARD_ROWS * BOARD_COLS, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, BOARD_ROWS * BOARD_COLS)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.device = 'cuda'
        self.qnet = DQN().to(self.device)
        self.target_qnet = DQN().to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=0.001, weight_decay=1e-5)
        self.epsilon = EPSILON_START
        self.count = 0
        self.target_update = 10
        

    def act(self, state, isran=False):
        available_positions = np.argwhere(state == 0)  # Get available positions as coordinates
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
        if isran or random.random() < self.epsilon:
            return random.choice(available_positions.flatten())  # Randomly choose from available positions

        else: 
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.qnet(state).detach().cpu().numpy().flatten()
            valid_q_values = np.full_like(q_values, -1e6)  # Initialize with a large negative value
            valid_q_values[available_positions.flatten()] = q_values[available_positions.flatten()]
            best_action = valid_q_values.argmax().item()
            return best_action
        



    def update(self, transition_dict):
        if len(self.memory) < BATCH_SIZE:
            return
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
                    -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_qnet.load_state_dict(
                self.qnet.state_dict())  # 更新目标网络
        self.count += 1


def play_with_agent(agent):
    env = TicTacToeEnv()
    state = env.reset()
    done = False
    print(env.board)

    while not done:
        if env.playerSymbol == 1:  # Human's turn
            while True:
                x = int(input("Enter row (0-2): "))
                y = int(input("Enter col (0-2): "))
                if (x, y) in env.availablePositions():
                    break
                print("Invalid move! Try again.")
            state, _, done = env.step(x * BOARD_COLS + y)
            print(state)
        else:  # Agent's turn
            print(env.board)
            if env.availablePositions() != []:  # Check if there are valid moves
                action = agent.act(state, isran=False)  # Choose the best valid action
                state, _, done = env.step(action)
                print(f"Agent played: {action}")
            else:
                print("No available moves left!")
                done = True  # End the game if no moves are available

        print(env.board)

        if done:
            reward, _ = env.check_winner()
            if reward == 1 and -env.playerSymbol == -1:
                print("Agent wins!")
            elif reward == 0.5:
                print("It's a draw!")
            else:
                print("You win!")
                
                
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()
    
    
def play_with_random(model):
    env = TicTacToeEnv()
    state = env.reset()
    agent = Agent()
    agent_random = Agent()
    total_reward = 0
    for i in range(500):
        state = env.reset()
        done = False
        
        while not done:
            if env.playerSymbol == 1:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                if reward == 0.5:
                    reward = 0
                total_reward += reward
                state = next_state
            else:
                action = agent_random.act(state, isran=True)
                next_state, reward, done = env.step(action)
                state = next_state
    print(total_reward/500)
    
    
from tqdm import tqdm
if __name__ == "__main__":
    env = TicTacToeEnv()

    replay_buffer1 = ReplayBuffer(MEMORY_SIZE)
    replay_buffer2 = ReplayBuffer(MEMORY_SIZE)
    agent1 = Agent()
    agent2 = Agent()
    
    return_list = []
    play_with_random(agent1)
    for i in range(10):
        with tqdm(total=int(50000 / 10), desc='Iteration %d' % i) as pbar:
            play_with_random(agent1)
            for e in range(5000):
                episode_return = 0
                state = env.reset()
                done = False
                total_reward = 0
                while not done:
                    if env.playerSymbol == 1:
                        action = agent1.act(state)
                        next_state, reward, done = env.step(action)
                        replay_buffer1.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        if replay_buffer1.size() > 500:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer1.sample(BATCH_SIZE)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d
                            }
                            
                            agent1.update(transition_dict)
                    else:
                        # action = agent2.act(state)
                        # next_state, reward, done = env.step(action)
                        # replay_buffer2.add(state, action, reward, next_state, done)
                        # state = next_state
                        # if replay_buffer2.size() > 500:
                        #     b_s, b_a, b_r, b_ns, b_d = replay_buffer2.sample(BATCH_SIZE)
                        #     transition_dict = {
                        #         'states': b_s,
                        #         'actions': b_a,
                        #         'next_states': b_ns,
                        #         'rewards': b_r,
                        #         'dones': b_d
                        #     }
                        #     agent2.update(transition_dict)
                        action = agent2.act(state, isran=True)
                        next_state, reward, done = env.step(action)
                        state = next_state

                return_list.append(episode_return)
                if (e + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (1000 / 10 * i + e + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
                
                
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format('ttt'))
    plt.show()
    torch.save(agent1.qnet.state_dict(), 'agent1.pth')
    play_with_agent(agent1)