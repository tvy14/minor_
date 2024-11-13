import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN

# Create the environment
env = gym.make('CartPole-v1')

# Define a function to train and evaluate an agent
def train_and_evaluate(agent_class, env, timesteps=10000):
    # Initialize the agent
    agent = agent_class('MlpPolicy', env, verbose=0)
    
    # Train the agent
    agent.learn(total_timesteps=timesteps)
    
    # Evaluate the agent
    obs = env.reset()
    rewards = []
    for _ in range(1000):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            obs = env.reset()
    
    return np.sum(rewards)

# Train and evaluate PPO
ppo_rewards = train_and_evaluate(PPO, env)

# Train and evaluate DQN
dqn_rewards = train_and_evaluate(DQN, env)

# Visualize the results
labels = ['PPO', 'DQN']
rewards = [ppo_rewards, dqn_rewards]

plt.bar(labels, rewards, color=['blue', 'green'])
plt.title('PPO vs DQN Performance on CartPole')
plt.xlabel('Algorithm')
plt.ylabel('Total Reward')
plt.show()