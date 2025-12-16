# -*- coding: utf-8 -*-
"""
Enhanced Tic Tac Toe using Reinforcement Learning
"""

import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1  # 🔁 FIXED: valid starting symbol

    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def winner(self):
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1

        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1

        diag1 = sum(self.board[i, i] for i in range(BOARD_COLS))
        diag2 = sum(self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS))
        if abs(diag1) == 3 or abs(diag2) == 3:
            self.isEnd = True
            return 1 if diag1 == 3 or diag2 == 3 else -1

        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0

        return None

    def availablePositions(self):
        return [(i, j) for i in range(BOARD_ROWS)
                        for j in range(BOARD_COLS) if self.board[i, j] == 0]

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol *= -1

    def giveReward(self):
        result = self.winner()
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(-1)   # 🔁 NEGATIVE reward for losing
        elif result == -1:
            self.p1.feedReward(-1)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.3)  # 🔁 higher draw reward
            self.p2.feedReward(0.3)

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=20000):  # 🔁 reduced rounds
        for i in range(rounds):
            while not self.isEnd:
                positions = self.availablePositions()
                action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(action)
                self.p1.addState(self.getHash())

                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                positions = self.availablePositions()
                action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(action)
                self.p2.addState(self.getHash())

                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

            if i % 5000 == 0:
                print(f"Training round {i}")

    def playWithHuman(self):
        while not self.isEnd:
            action = self.p1.chooseAction(self.availablePositions(), self.board, self.playerSymbol)
            self.updateState(action)
            self.showBoard()

            if self.winner() is not None:
                print("Computer wins or Draw!")
                self.reset()
                break

            action = self.p2.chooseAction(self.availablePositions())
            self.updateState(action)
            self.showBoard()

            if self.winner() is not None:
                print("Human wins or Draw!")
                self.reset()
                break

    def showBoard(self):
        print("\nCurrent Board:")
        for i in range(BOARD_ROWS):
            row = ""
            for j in range(BOARD_COLS):
                if self.board[i, j] == 1:
                    row += " X "
                elif self.board[i, j] == -1:
                    row += " O "
                else:
                    row += " _ "
            print(row)
        print()


class Player:
    def __init__(self, name, exp_rate=0.1):  # 🔁 reduced exploration
        self.name = name
        self.states = []
        self.lr = 0.3        # 🔁 increased learning rate
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}

    def getHash(self, board):
        return str(board.reshape(BOARD_COLS * BOARD_ROWS))

    def chooseAction(self, positions, current_board, symbol):
        if np.random.rand() < self.exp_rate:
            return positions[np.random.choice(len(positions))]

        value_max = -float('inf')
        for p in positions:
            next_board = current_board.copy()
            next_board[p] = symbol
            value = self.states_value.get(self.getHash(next_board), 0)
            if value >= value_max:
                value_max = value
                action = p
        return action

    def addState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        for st in reversed(self.states):
            self.states_value[st] = self.states_value.get(st, 0) + \
                self.lr * (self.decay_gamma * reward - self.states_value.get(st, 0))
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        with open('policy_' + self.name, 'wb') as f:
            pickle.dump(self.states_value, f)

    def loadPolicy(self, file):
        with open(file, 'rb') as f:
            self.states_value = pickle.load(f)


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            try:
                row = int(input("Row (0-2): "))
                col = int(input("Col (0-2): "))
                if (row, col) in positions:
                    return (row, col)
            except:
                pass
            print("Invalid move. Try again.")


if __name__ == "__main__":
    p1 = Player("AI")
    p2 = Player("AI_Trainer")

    state = State(p1, p2)
    print("Training AI...")
    state.play()

    p1.savePolicy()

    p1 = Player("Computer", exp_rate=0)
    p1.loadPolicy("policy_AI")

    human = HumanPlayer("Human")
    game = State(p1, human)

    while True:
        game.playWithHuman()
        if input("Play again? (y/n): ").lower() != 'y':
            break
