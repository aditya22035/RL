import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create environment
env = gym.make('CliffWalking-v0')

# Function to convert state integer to one-hot encoded tensor
def state_to_tensor(state):
    # CliffWalking has 48 states (4x12 grid)
    one_hot = np.zeros(48, dtype=np.float32)
    one_hot[state] = 1.0
    return torch.FloatTensor(one_hot).unsqueeze(0)

# PPO Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        with torch.no_grad():
            state_tensor = state_to_tensor(state)
            action_probs, _ = self(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            return action.item(), action_probs[0, action.item()].item()

# PPO Agent
class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon_clip = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.learning_rate = 0.0003
        self.k_epochs = 4  # Number of optimization epochs
        
        # Networks
        self.policy = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Memory for batch update
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        # For tracking performance
        self.best_reward = -float('inf')
    
    def reset_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def select_action(self, state):
        action, log_prob = self.policy.act(state)
        
        # Get value estimate
        with torch.no_grad():
            state_tensor = state_to_tensor(state)
            _, value = self.policy(state_tensor)
        
        # Store in memory
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value.item())
        
        return action
    
    def update_memory(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_returns(self, final_value=0):
        returns = []
        R = final_value
        
        # Calculate discounted returns
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + self.gamma * R * (1 - self.dones[step])
            returns.insert(0, R)
            
        # Normalize returns for stable training
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
        return returns
    
    def update(self):
        # Convert lists to tensors
        states_tensor = torch.cat([state_to_tensor(s) for s in self.states])
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.long)
        
        # Compute returns
        final_value = 0
        if len(self.states) > 0 and not self.dones[-1]:
            with torch.no_grad():
                _, final_value = self.policy(state_to_tensor(self.states[-1]))
                final_value = final_value.item()
                
        returns = self.compute_returns(final_value)
        
        # Update policy for K epochs
        for _ in range(self.k_epochs):
            # Get current action probabilities and state values
            action_probs, state_values = self.policy(states_tensor)
            dist = Categorical(action_probs)
            curr_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1))
            entropy = dist.entropy().mean()
            
            # Calculate advantage
            advantages = returns - state_values.squeeze()
            
            # Calculate ratios
            ratios = torch.exp(curr_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages
            
            # Calculate final loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.value_coef * nn.MSELoss()(state_values.squeeze(), returns)
            entropy_loss = -self.entropy_coef * entropy
            
            loss = actor_loss + critic_loss + entropy_loss
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients for stability
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Reset memory
        self.reset_memory()
        
        return loss.item()
    
    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)
        
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))

# Training the PPO agent
def train_ppo(env, agent, num_episodes=500, update_timestep=2048, print_interval=25):
    rewards = []
    timestep = 0
    best_avg_reward = -float('inf')
    
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
            
            # Update memory
            agent.update_memory(reward, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            timestep += 1
            
            # Update policy if it's time
            if timestep % update_timestep == 0:
                agent.update()
        
        # Update at the end of episode if there's data
        if len(agent.states) > 0:
            agent.update()
        
        # Store rewards
        rewards.append(episode_reward)
        
        # Print progress
        if episode % print_interval == 0:
            avg_reward = np.mean(rewards[-print_interval:])
            print(f"Episode {episode}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model('ppo_best_model.pth')
    
    return rewards

# Plot training rewards
def plot_rewards(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('PPO Training Rewards')
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
    path = []
    
    while not done:
        path.append(state)
        r, c = state // cols, state % cols
        
        # Don't mark start and goal with dots
        if state != start and state != goal:
            plt.text(c, r, 'o', ha='center', color='red')
        
        # Use the policy to select the best action
        action, _ = agent.policy.act(state)
        
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
    
    plt.title("Cliff Walking Path (PPO)")
    plt.show()
    
    return path

# Compare algorithm performances
def compare_algorithms(dqn_rewards, ppo_rewards):
    plt.figure(figsize=(12, 6))
    
    # Apply smoothing for readability
    def smooth(rewards, window=10):
        return np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    smooth_dqn = smooth(dqn_rewards)
    smooth_ppo = smooth(ppo_rewards)
    
    plt.plot(smooth_dqn, label='DQN')
    plt.plot(smooth_ppo, label='PPO')
    plt.title('Algorithm Comparison: DQN vs PPO')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print final performance statistics
    print(f"DQN Final Average Reward (last 50 episodes): {np.mean(dqn_rewards[-50:]):.2f}")
    print(f"PPO Final Average Reward (last 50 episodes): {np.mean(ppo_rewards[-50:]):.2f}")
    improvement = (np.mean(ppo_rewards[-50:]) - np.mean(dqn_rewards[-50:])) / abs(np.mean(dqn_rewards[-50:])) * 100
    print(f"Improvement: {improvement:.2f}%")

# Main function
def main():
    # Environment parameters
    state_size = 48  # CliffWalking has 48 states (4x12 grid)
    action_size = env.action_space.n  # 4 actions (up, right, down, left)
    
    # Initialize PPO agent
    ppo_agent = PPOAgent(state_size, action_size)
    
    # Train PPO agent
    print("Training PPO agent...")
    ppo_rewards = train_ppo(env, ppo_agent, num_episodes=500)
    
    # Plot PPO training rewards
    plot_rewards(ppo_rewards)
    
    # Load best model
    ppo_agent.load_model('ppo_best_model.pth')
    
    # Plot optimal path
    print("Plotting optimal path using PPO agent...")
    path = plot_cliff_path(env, ppo_agent)
    print(f"Path length: {len(path)}")
    
    # Test with DQN from previous implementation
    try:
        from cliff_walking_dqn import DQNAgent, train_dqn
        print("Comparing with DQN...")
        
        # Initialize DQN agent
        dqn_agent = DQNAgent(state_size, action_size)
        
        # Train DQN agent
        dqn_rewards = train_dqn(env, dqn_agent, num_episodes=500)
        
        # Compare algorithms
        compare_algorithms(dqn_rewards, ppo_rewards)
    except ImportError:
        print("DQN implementation not available for comparison.")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
