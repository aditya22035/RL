import numpy as np  
import matplotlib.pyplot as plt
import gymnasium as gym
import time
import os
from collections import defaultdict

def epsilon_greedy_policy(epsilon, Q, env):
    def policy(state):
        if state not in Q:
            return np.random.choice(env.action_space.n)
        else:
            if np.random.random() > epsilon:
                return np.argmax(Q[state])
            else:
                return np.random.choice(env.action_space.n)
    return policy

def sarsa_nstep(env, numeps, epsilon, alpha, gamma, nstep):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for i in range(numeps):
        if i % 500 == 0:
            print("Episode: ", i)
            
        state, _ = env.reset()
        done = False
        T = float('inf')
        t = 0
        
        # Initialize first action
        policy = epsilon_greedy_policy(epsilon, Q, env)
        action = policy(state)
        
        buffer = []  # stores state, action, reward tuples
        
        while True:
            if t < T:
                # Take step and store transition
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.append((state, action, reward))
                
                if done:
                    T = t + 1
                else:
                    state = next_state
                    action = policy(state)
            
            # Calculate update time
            tau = t - nstep + 1
            
            if tau >= 0:
                # Calculate return
                G = 0
                for k in range(tau+1, min(tau+nstep, T)+1):
                    G += (gamma**(k-tau-1)) * buffer[k-1][2]  # Use reward from buffer
                
                # Add bootstrap value if not at terminal state
                if tau + nstep < T:
                    state_tau_n = buffer[tau+nstep-1][0]  # Get state from buffer
                    action_tau_n = buffer[tau+nstep-1][1]  # Get action from buffer
                    G += (gamma**nstep) * Q[state_tau_n][action_tau_n]
                
                # Update Q-value
                state_tau = buffer[tau][0]
                action_tau = buffer[tau][1]
                Q[state_tau][action_tau] += alpha * (G - Q[state_tau][action_tau])
            
            t += 1
            
            if tau == T - 1:
                break
                
    return Q, policy


def plot_cliffwalking_paths(Q, env):
    grid_rows, grid_cols = 4, 12  # Grid dimensions for CliffWalking-v0
    start_state = 36  # Start state index
    goal_state = 47   # Goal state index

    # Initialize the grid
    grid = np.zeros((grid_rows, grid_cols), dtype=int)

    # Define the cliff area
    cliff_indices = np.arange(37, 47)
    for idx in cliff_indices:
        row, col = divmod(idx, grid_cols)
        grid[row, col] = -100  # Cliff cells

    # Compute the optimal path
    state, _ = env.reset()
    optimal_path = []
    done = False
    while not done:
        row, col = divmod(state, grid_cols)
        optimal_path.append((row, col))
        action = np.argmax(Q[state]) if state in Q else 0
        tup = env.step(action)
        state, _, done, _ = tup if len(tup) == 4 else (tup[0], tup[1], tup[2], None)

    # Mark the start and goal positions
    start_row, start_col = divmod(start_state, grid_cols)
    goal_row, goal_col = divmod(goal_state, grid_cols)

    # Plot the grid
    plt.figure(figsize=(12, 4))
    for r in range(grid_rows):
        for c in range(grid_cols):
            if (r, c) in optimal_path:
                color = "red" if (r, c) != (start_row, start_col) and (r, c) != (goal_row, goal_col) else "green"
                plt.text(c, r, 'O', ha='center', va='center', color=color, fontsize=12, fontweight='bold')
            elif grid[r, c] == -100:
                plt.text(c, r, 'Cliff', ha='center', va='center', color='gray', fontsize=8, fontweight='bold')

    # Highlight start and goal
    plt.text(start_col, start_row, 'S', ha='center', va='center', color='blue', fontsize=14, fontweight='bold')
    plt.text(goal_col, goal_row, 'G', ha='center', va='center', color='blue', fontsize=14, fontweight='bold')

    # Draw the grid
    plt.xlim(-0.5, grid_cols - 0.5)
    plt.ylim(grid_rows - 0.5, -0.5)
    plt.xticks(range(grid_cols))
    plt.yticks(range(grid_rows))
    plt.grid(True)
    plt.title("Cliff Walking Optimal Path")
    plt.show()

# Plot the optimal path



if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    numeps = 1000
    epsilon = 0.1
    alpha = 0.1
    nsteps = 5
    gamma = 1
    
    
    
    Q, policy = sarsa_nstep(env, numeps, epsilon, alpha, gamma, nsteps)
    plot_cliffwalking_paths(Q, env)
    