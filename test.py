import numpy as np
from collections import defaultdict
import gymnasium as gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def observation_to_tuple(observation):
    """Convert observation into a clean tuple."""
    return observation[0], observation[1], observation[2]

def print_observation(observation):
    """Print details of an observation."""
    player_score, dealer_score, usable_ace = observation
    print(f"Player Score: {player_score} (Usable Ace: {usable_ace}), Dealer Score: {dealer_score}")

def initialize_random_policy():
    """Initialize a random policy for the Blackjack environment."""
    return lambda state: np.random.choice([0, 1])

def generate_episode(env, policy):
    """Generate an episode using the provided policy."""
    state, _ = env.reset()
    episode = []
    start = True

    while True:
        if start:
            action = np.random.choice(env.action_space.n)  # Exploring start
            start = False
        else:
            action = policy(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))

        if terminated or truncated:
            break

        state = next_state

    return episode

def update_q_values(episode, Q, returns_sum, returns_count):
    """Update Q-values based on the episode."""
    state_action_pairs = [(state, action) for state, action, _ in episode]
    for idx, (state, action) in enumerate(state_action_pairs):
        # Check if this is the first occurrence of the state-action pair in the episode
        if (state, action) not in state_action_pairs[:idx]:
            # Calculate the total return from this point onward
            G = sum([reward for _, _, reward in episode[idx:]])

            # Update the sum of returns and count fo    r the state-action pair
            returns_sum[(state, action)] += G
            returns_count[(state, action)] += 1

            # Update the Q-value
            Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

def create_greedy_policy(Q):
    """Create a greedy policy based on the Q-values."""
    def policy(state):
        if state not in Q:
            return np.random.choice([0, 1])  # Default action if state is unseen
        return np.argmax(Q[state])

    return policy

def mc_control_with_exploring_starts(env, num_episodes, debug=False):
    """Monte Carlo control with exploring starts for Blackjack."""
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for episode_num in range(1, num_episodes + 1):
        if debug and episode_num % 10000 == 0:
            print(f"Episode {episode_num}/{num_episodes}")

        # Generate an episode
        policy = initialize_random_policy()
        episode = generate_episode(env, policy)

        # Update Q-values
        update_q_values(episode, Q, returns_sum, returns_count)

    # Derive the optimal policy from Q-values
    optimal_policy = create_greedy_policy(Q)
    return Q, optimal_policy

def plot_3d_value_function(V):
    """Plot the 3D value function for states without a usable ace."""
    player_scores = np.arange(12, 22)
    dealer_scores = np.arange(1, 11)
    values = np.zeros((len(player_scores), len(dealer_scores)))

    for i, player in enumerate(player_scores):
        for j, dealer in enumerate(dealer_scores):
            state = (player, dealer, False)  # Usable ace is False
            if state in V:
                values[i, j] = V[state]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(dealer_scores, player_scores)
    ax.plot_surface(X, Y, values, cmap='viridis')
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Score')
    ax.set_zlabel('Value')
    ax.set_title('Value Function (No Usable Ace)')
    plt.show()

if __name__ == "__main__":
    env = gym.make("Blackjack-v1", sab=True)  # Ensure you're using Gymnasium-compatible environment

    # Run Monte Carlo Control with Exploring Starts
    num_episodes = 500000
    Q, optimal_policy = mc_control_with_exploring_starts(env, num_episodes, debug=True)

    # Compute state-value function from action-value function
    V = {state: max(actions) for state, actions in Q.items()}

            # # Example usage of the optimal policy
            # test_state = (18, 10, False)
            # print("Optimal action for state", test_state, ":", optimal_policy(test_state))

    # Plot the 3D value function
    plot_3d_value_function(V)
