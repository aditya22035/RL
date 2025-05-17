import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create environment
env = gym.make('CliffWalking-v0')

# Define DQN network architecture
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Function to convert state integer to one-hot encoded tensor
def state_to_tensor(state):
    # CliffWalking has 48 states (4x12 grid)
    one_hot = np.zeros(48, dtype=np.float32)
    one_hot[state] = 1.0
    return torch.FloatTensor(one_hot).unsqueeze(0)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural networks
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay memory
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        
        # For saving best model
        self.best_reward = -float('inf')
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()
    
    def step(self, state, action, reward, next_state, done):
        # Save to replay memory
        self.memory.push(state, action, reward, next_state, done)
        
        # Learn if enough samples are available
        if len(self.memory) > self.batch_size:
            self.learn()
    
    def learn(self):
        # Sample batch from replay memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_tensors = torch.cat([state_to_tensor(s) for s in states])
        action_tensors = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        reward_tensors = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_state_tensors = torch.cat([state_to_tensor(s) for s in next_states])
        done_tensors = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        # Get current Q values
        current_q_values = self.policy_net(state_tensors).gather(1, action_tensors)
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_tensors).max(1)[0].unsqueeze(1)
        
        # Compute target Q values
        target_q_values = reward_tensors + (self.gamma * next_q_values * (1 - done_tensors))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Training the DQN agent
def train_dqn(env, agent, num_episodes=500, target_update=10, print_interval=25):
    rewards = []
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update agent
            agent.step(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Update target network
        if episode % target_update == 0:
            agent.update_target_net()
        
        # Store rewards
        rewards.append(episode_reward)
        
        # Print progress
        if episode % print_interval == 0:
            avg_reward = np.mean(rewards[-print_interval:])
            print(f"Episode {episode}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return rewards

# Plot training rewards
def plot_rewards(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('DQN Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

# Plot cliff walking path
def plot_cliff_path(env, agent):
    # Grid dimensions
    rows, cols = 4, 12
    start, goal = 36, 47
    
    # Setup plot
    plt.figure(figsize=(12, 4))
    
    # Draw grid
    plt.grid(True)
    plt.xlim(-0.5, cols - 0.5)
    plt.ylim(rows - 0.5, -0.5)
    plt.xticks(range(cols))
    plt.yticks(range(rows))
    
    # Draw cliffs
    for cliff in range(37, 47):
        r, c = cliff // cols, cliff % cols
        plt.text(c, r, 'C', ha='center', color='gray')
    
    # Draw start and goal
    plt.text(start % cols, start // cols, 'S', ha='center', color='blue')
    plt.text(goal % cols, goal // cols, 'G', ha='center', color='blue')
    
    # Draw optimal path
    state, _ = env.reset()
    done = False
    
    while not done:
        r, c = state // cols, state % cols
        
        # Don't mark start and goal with dots
        if state != start and state != goal:
            plt.text(c, r, 'o', ha='center', color='red')
        
        # Select action using policy network
        with torch.no_grad():
            state_tensor = state_to_tensor(state)
            q_values = agent.policy_net(state_tensor)
            action = torch.argmax(q_values).item()
        
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
    
    plt.title("Cliff Walking Path (DQN)")
    plt.show()

# Main function
def main():
    # Environment parameters
    state_size = 48  # CliffWalking has 48 states (4x12 grid)
    action_size = env.action_space.n  # 4 actions (up, right, down, left)
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size)
    
    # Train agent
    print("Training DQN agent...")
    rewards = train_dqn(env, agent, num_episodes=500)
    
    # Plot training rewards
    plot_rewards(rewards)
    
    # Plot optimal path
    plot_cliff_path(env, agent)
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
