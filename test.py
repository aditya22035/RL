import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict
import copy

def epsilon_greedy(Q, epsilon, action_space_n):
    def policy(state):
        if state not in Q:
            return np.random.choice(action_space_n)
        # Vectorized operation for epsilon-greedy
        if np.random.random() < epsilon:
            return np.random.choice(action_space_n)
        return int(np.argmax(Q[state]))
    return policy

def sumQ(Q1, Q2):
    # More efficient dictionary merging
    Q = defaultdict(lambda: np.zeros(Q1[next(iter(Q1))].shape))
    for key in set(Q1.keys()) | set(Q2.keys()):
        Q[key] = Q1[key] + Q2[key]
    return Q

def double_qlearning(env, numeps, epsilon, alpha, gamma):
    action_space_n = env.action_space.n
    # Pre-allocate numpy arrays for Q-values
    Q1 = defaultdict(lambda: np.zeros(action_space_n, dtype=np.float32))
    Q2 = defaultdict(lambda: np.zeros(action_space_n, dtype=np.float32))
    
    # Create policies once
    policy = epsilon_greedy(sumQ(Q1, Q2), epsilon, action_space_n)
    
    # Vectorized operations for Q-value updates
    for i in range(1, numeps + 1):
        if i % 1000 == 0:
            print(f'episode = {i}')
        
        state, _ = env.reset()
        done = False
        
        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Vectorized Q-value updates
            if np.random.random() < 0.5:
                action_greedy = np.argmax(Q1[next_state])
                Q1[state][action] += alpha * (reward + gamma * Q2[next_state][action_greedy] - Q1[state][action])
            else:
                action_greedy = np.argmax(Q2[next_state])
                Q2[state][action] += alpha * (reward + gamma * Q1[next_state][action_greedy] - Q2[state][action])
            
            state = next_state
    
    final_Q = sumQ(Q1, Q2)
    return lambda s: np.argmax(final_Q[s]), final_Q

def plot_cliffwalking_paths(Q, env, policy):
    grid_rows, grid_cols = 4, 12
    start_state, goal_state = 36, 47
    
    # Pre-allocate arrays
    grid = np.zeros((grid_rows, grid_cols), dtype=np.int8)
    cliff_indices = np.arange(37, 47)
    optimal_path = []
    
    # Vectorized cliff marking
    rows, cols = np.divmod(cliff_indices, grid_cols)
    grid[rows, cols] = -100
    
    # Compute optimal path
    state, _ = env.reset()
    done = False
    while not done:
        row, col = divmod(state, grid_cols)
        optimal_path.append((row, col))
        action = policy(state)
        next_state, reward, terminated, truncated, _= env.step(action)
        done = terminated or truncated
        state = next_state
    
    # Plot optimization
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Batch plot operations
    for r, c in optimal_path:
        if (r, c) != divmod(start_state, grid_cols) and (r, c) != divmod(goal_state, grid_cols):
            ax.text(c, r, 'O', ha='center', va='center', color='red', fontsize=12, fontweight='bold')
    
    # Vectorized cliff plotting
    for r, c in zip(rows, cols):
        ax.text(c, r, 'Cliff', ha='center', va='center', color='gray', fontsize=8, fontweight='bold')
    
    # Plot start and goal
    start_row, start_col = divmod(start_state, grid_cols)
    goal_row, goal_col = divmod(goal_state, grid_cols)
    ax.text(start_col, start_row, 'S', ha='center', va='center', color='blue', fontsize=14, fontweight='bold')
    ax.text(goal_col, goal_row, 'G', ha='center', va='center', color='blue', fontsize=14, fontweight='bold')
    
    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(grid_rows - 0.5, -0.5)
    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.grid(True)
    ax.set_title("Cliff Walking Optimal Path")
    plt.show()

# Usage
env = gym.make('CliffWalking-v0')
numeps = 1000
epsilon = 0.1
alpha = 0.1
gamma = 1.0

policy, Q = double_qlearning(env, numeps, epsilon, alpha, gamma)
print("reached")
plot_cliffwalking_paths(Q, env, policy)