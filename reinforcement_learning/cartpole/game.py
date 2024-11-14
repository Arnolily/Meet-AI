import gym
import pygame
import time
from pygame.locals import *

# Initialize the CartPole environment with a modified time limit
env = gym.make('CartPole-v1', render_mode='human')
env._max_episode_steps = 1000  # Increase max steps to prevent early termination
env.reset()

# Initialize pygame for keyboard control
pygame.init()
win = pygame.display.set_mode((400, 300))
pygame.display.set_caption("CartPole Manual Control")

def play_cartpole():
    done = False
    action = 0  # Initialize action to move left
    total_reward = 0
    while not done:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                env.close()
                return
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    action = 0  # Move left
                elif event.key == K_RIGHT:
                    action = 1  # Move right
            elif event.type == KEYUP:
                action = None  # Stop movement when no key is pressed

        if action is not None:
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()
            print(f"Obs: {obs}, Reward: {reward}")  # Display state and reward for tracking

        time.sleep(0.05)  # Add a delay to slow down the game

        if done:
            print("Game Over. Restarting...")
            print(f'total reward:{total_reward}')
            env.reset()

# Start the game
try:
    play_cartpole()
finally:
    env.close()
    pygame.quit()
