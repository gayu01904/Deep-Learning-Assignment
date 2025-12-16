import numpy as np
import pylab as pl
import random

# ---------------- ENVIRONMENT ----------------
edges = [
    (0, 1), (1, 5), (5, 6), (5, 4), (1, 2),
    (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
    (8, 9), (7, 8), (1, 7), (3, 9)
]

goal = 10
MATRIX_SIZE = 11

gamma = 0.8
alpha = 0.6
epsilon = 0.4

# ---------------- REWARD MATRIX ----------------
M = np.full((MATRIX_SIZE, MATRIX_SIZE), -10)

for (i, j) in edges:
    M[i, j] = 100 if j == goal else 0
    M[j, i] = 100 if i == goal else 0

M[goal, goal] = 100

# ---------------- Q MATRIX ----------------
Q = np.zeros((MATRIX_SIZE, MATRIX_SIZE))

# ---------------- FUNCTIONS ----------------
def available_actions(state):
    return np.where(M[state] >= 0)[0]

def choose_action(state):
    actions = available_actions(state)
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(Q[state, actions])]

def update_q(state, action):
    best_next = np.max(Q[action])
    Q[state, action] += alpha * (M[state, action] + gamma * best_next - Q[state, action])

# ---------------- TRAINING ----------------
scores = []

for i in range(1200):
    state = random.randint(0, MATRIX_SIZE - 1)
    action = choose_action(state)
    update_q(state, action)

    scores.append(np.sum(Q))
    epsilon = max(0.05, epsilon * 0.995)

# ---------------- TESTING ----------------
current_state = 0
path = [current_state]

while current_state != goal:
    next_state = np.argmax(Q[current_state])
    path.append(next_state)
    current_state = next_state

print("Optimal path to goal:")
print(path)

# ---------------- PLOT ----------------
pl.plot(scores)
pl.xlabel("Iterations")
pl.ylabel("Cumulative Q-value")
pl.title("Q-Learning Convergence")
pl.show()
