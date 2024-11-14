import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gym
import pygame
import sys
class TicTacToeEnv:
    def __init__(self):
        pygame.init()
        self.size = 300
        self.line_width = 15
        self.board_rows = 3
        self.board_cols = 3
        self.square_size = self.size // self.board_rows
        self.circle_radius = self.square_size // 3
        self.circle_width = 15
        self.cross_width = 25
        self.space = self.square_size // 4
        self.screen = pygame.display.set_mode((self.size, self.size))
        pygame.display.set_caption('Tic Tac Toe')
        self.screen.fill((28, 170, 156))
        self.board = np.zeros((self.board_rows, self.board_cols))
        self.draw_lines()
        self.actions = []
        self.states = []
        self.rewards = []

    def draw_lines(self):
        for row in range(1, self.board_rows):
            pygame.draw.line(self.screen, (23, 145, 135), (0, row * self.square_size), (self.size, row * self.square_size), self.line_width)
            pygame.draw.line(self.screen, (23, 145, 135), (row * self.square_size, 0), (row * self.square_size, self.size), self.line_width)

    def draw_figures(self):
        for row in range(self.board_rows):
            for col in range(self.board_cols):
                if self.board[row][col] == 1:
                    pygame.draw.circle(self.screen, (239, 231, 200), (int(col * self.square_size + self.square_size // 2), int(row * self.square_size + self.square_size // 2)), self.circle_radius, self.circle_width)
                elif self.board[row][col] == 2:
                    pygame.draw.line(self.screen, (66, 66, 66), (col * self.square_size + self.space, row * self.square_size + self.square_size - self.space), (col * self.square_size + self.square_size - self.space, row * self.square_size + self.space), self.cross_width)
                    pygame.draw.line(self.screen, (66, 66, 66), (col * self.square_size + self.space, row * self.square_size + self.space), (col * self.square_size + self.square_size - self.space, row * self.square_size + self.square_size - self.space), self.cross_width)

    def mark_square(self, row, col, player):
        self.board[row][col] = player
        self.actions.append((row, col, player))
        self.states.append(self.board.copy())
        reward = 1 if self.check_win(player) else 0
        self.rewards.append(reward)

    def available_square(self, row, col):
        return self.board[row][col] == 0

    def is_board_full(self):
        for row in range(self.board_rows):
            for col in range(self.board_cols):
                if self.board[row][col] == 0:
                    return False
        return True

    def check_win(self, player):
        for col in range(self.board_cols):
            if self.board[0][col] == player and self.board[1][col] == player and self.board[2][col] == player:
                return True
        for row in range(self.board_rows):
            if self.board[row][0] == player and self.board[row][1] == player and self.board[row][2] == player:
                return True
        if self.board[0][0] == player and self.board[1][1] == player and self.board[2][2] == player:
            return True
        if self.board[2][0] == player and self.board[1][1] == player and self.board[0][2] == player:
            return True
        return False

    def restart(self):
        self.screen.fill((28, 170, 156))
        self.draw_lines()
        self.board = np.zeros((self.board_rows, self.board_cols))
        self.actions = []
        self.states = []
        self.rewards = []

    def step(self, row, col, player):
        if self.available_square(row, col):
            self.mark_square(row, col, player)
            self.draw_figures()
            reward = 1 if self.check_win(player) else 0
            done = self.check_win(player) or self.is_board_full()
            return self.board.copy(), reward, done
        else:
            return self.board.copy(), -1, False

def main():
    env = TicTacToeEnv()
    player = 1
    game_over = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                mouseX = event.pos[0]
                mouseY = event.pos[1]
                clicked_row = int(mouseY // env.square_size)
                clicked_col = int(mouseX // env.square_size)

                if env.available_square(clicked_row, clicked_col):
                    env.mark_square(clicked_row, clicked_col, player)
                    if env.check_win(player):
                        game_over = True
                        print(f'Player {player} wins!')
                    player = player % 2 + 1
                    env.draw_figures()

        pygame.display.update()

if __name__ == "__main__":
    main()
