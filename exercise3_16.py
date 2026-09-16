# Exercise 3.16
# Design your own 8-state routing diagram and adapt Example 3.22's Q-learning
# program to solve it.
#
# Note: Example 3.22 in the book relies on a helper module (Q_Utils.py) not
# included here. This is a self-contained rewrite of the same idea (the
# standard pattern used in most Q-learning routing tutorials).
import numpy as np

# Custom 8-state routing diagram (states 0-7), goal state = 7
points_list = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)]
goal = 7
MATRIX_SIZE = 8

# R: -1 = no direct link, 0 = link (not to goal), 100 = link to goal
R = np.matrix(np.ones((MATRIX_SIZE, MATRIX_SIZE))) * -1
for a, b in points_list:
    R[a, b] = 100 if b == goal else 0
    R[b, a] = 100 if a == goal else 0
R[goal, goal] = 100

Q = np.matrix(np.zeros((MATRIX_SIZE, MATRIX_SIZE)))
gamma = 0.8

def available_actions(state):
    return np.where(np.array(R[state])[0] >= 0)[0]

def sample_next_action(available_act):
    return int(np.random.choice(available_act))

def update(current_state, action):
    max_value = np.max(Q[action])
    Q[current_state, action] = R[current_state, action] + gamma * max_value

np.random.seed(0)
for i in range(1000):
    current_state = np.random.randint(0, MATRIX_SIZE)
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state, action)

print("Trained Q matrix (normalized):")
print(Q / np.max(Q) * 100)

current_state = 0
route = [current_state]
while current_state != goal:
    next_state = int(np.argmax(Q[current_state]))
    route.append(next_state)
    current_state = next_state
print("Best route from 0 to goal:", route)
