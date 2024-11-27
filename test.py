import numpy as np  
import matplotlib.pyplot as plt
import gymnasium as gym
import time
import os
from collections import defaultdict

env = gym.make('CliffWalking-v0')
numactions = env.action_space.n
numstates = env.observation_space.n

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

def get_action_probabilities(state, Q, epsilon, num_actions):
    """Calculate probabilities for each action under epsilon-greedy policy"""
    if state not in Q:
        # If state not seen before, return uniform distribution
        return np.ones(num_actions) / num_actions
    
    probs = np.ones(num_actions) * epsilon / num_actions  # exploration probabilities
    best_action = np.argmax(Q[state])
    probs[best_action] += 1 - epsilon  # add exploitation probability
    return probs

def expected_sarsa(env, numeps, epsilon, alpha, gamma):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    reward_history = np.zeros(numeps)
    
    for i in range(numeps):
        if i % 10000 == 0:
            print("Episode ", i)
            
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get current policy and action
            policy = epsilon_greedy_policy(epsilon, Q, env)
            action = policy(state)
            
            # Take action
            tup = env.step(action)
            next_state, reward, done = tup[0], tup[1], tup[2]
            
            # Calculate expected value of next state
            next_action_probs = get_action_probabilities(next_state, Q, epsilon, env.action_space.n)
            expected_value = np.sum(next_action_probs * Q[next_state])
            
            # Update Q-value using expected SARSA update rule
            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * expected_value - Q[state][action]
            )
            
            state = next_state
            episode_reward += reward
            
        reward_history[i] = episode_reward
        
    return Q, policy, reward_history

# Run the algorithm
Q, policy, reward_history = expected_sarsa(env, 1000, 0.1, 0.1, 1.0)

# Plotting functions remain the same
def plot_rewards(reward_history):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Expected SARSA Reward History")
    plt.show()

def plot_value_function(Q):
    V = defaultdict(float)
    for state, actions in Q.items():
        V[state] = np.max(actions)
    
    V = np.array([V[key] for key in np.arange(0, 48)])
    V = np.reshape(V, (4, 12))
    
    plt.figure(figsize=(12, 4))
    plt.imshow(V, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Expected SARSA Value Function")
    plt.show()

def plot_cliffwalking_paths(Q, env):
    grid_rows, grid_cols = 4, 12
    start_state = 36
    goal_state = 47

    grid = np.zeros((grid_rows, grid_cols), dtype=int)
    cliff_indices = np.arange(37, 47)
    for idx in cliff_indices:
        row, col = divmod(idx, grid_cols)
        grid[row, col] = -100

    state, _ = env.reset()
    optimal_path = []
    done = False
    while not done:
        row, col = divmod(state, grid_cols)
        optimal_path.append((row, col))
        action = np.argmax(Q[state])
        tup = env.step(action)
        state, _, done = tup[0], tup[1], tup[2]

    start_row, start_col = divmod(start_state, grid_cols)
    goal_row, goal_col = divmod(goal_state, grid_cols)

    plt.figure(figsize=(12, 4))
    for r in range(grid_rows):
        for c in range(grid_cols):
            if (r, c) in optimal_path:
                color = "red" if (r, c) != (start_row, start_col) and (r, c) != (goal_row, goal_col) else "green"
                plt.text(c, r, 'O', ha='center', va='center', color=color, fontsize=12, fontweight='bold')
            elif grid[r, c] == -100:
                plt.text(c, r, 'Cliff', ha='center', va='center', color='gray', fontsize=8, fontweight='bold')

    plt.text(start_col, start_row, 'S', ha='center', va='center', color='blue', fontsize=14, fontweight='bold')
    plt.text(goal_col, goal_row, 'G', ha='center', va='center', color='blue', fontsize=14, fontweight='bold')

    plt.xlim(-0.5, grid_cols - 0.5)
    plt.ylim(grid_rows - 0.5, -0.5)
    plt.xticks(range(grid_cols))
    plt.yticks(range(grid_rows))
    plt.grid(True)
    plt.title("Expected SARSA Optimal Path")
    plt.show()

# Generate all plots
plot_rewards(reward_history)
plot_value_function(Q)
plot_cliffwalking_paths(Q, env) 