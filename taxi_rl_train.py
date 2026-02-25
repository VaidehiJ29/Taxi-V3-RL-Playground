"""
Taxi-v3 Q-Learning Agent
Run this script to train the agent and export the Q-table.
Requirements: pip install gymnasium numpy
"""

import gymnasium as gym
import numpy as np
import json
import random

# ─── Hyperparameters ─────────────────────────────────────────────────────────
EPISODES        = 10_000
ALPHA           = 0.1      # Learning rate
GAMMA           = 0.99     # Discount factor
EPSILON_START   = 1.0
EPSILON_END     = 0.01
EPSILON_DECAY   = 0.0005
# ─────────────────────────────────────────────────────────────────────────────

env = gym.make("Taxi-v3")
n_states  = env.observation_space.n   # 500
n_actions = env.action_space.n        # 6

Q = np.zeros((n_states, n_actions))

epsilon = EPSILON_START
rewards_history = []

print("Training Taxi-v3 Q-Learning Agent...")
for episode in range(1, EPISODES + 1):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # ε-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-update (Bellman equation)
        best_next = np.max(Q[next_state])
        Q[state, action] += ALPHA * (reward + GAMMA * best_next - Q[state, action])

        state = next_state
        total_reward += reward

    epsilon = max(EPSILON_END, epsilon - EPSILON_DECAY)
    rewards_history.append(total_reward)

    if episode % 1000 == 0:
        avg = np.mean(rewards_history[-1000:])
        print(f"  Episode {episode:>6} | Avg Reward (last 1k): {avg:+.2f} | ε: {epsilon:.4f}")

env.close()

# ─── Export Q-table as JSON for the web UI ───────────────────────────────────
q_list = Q.tolist()
with open("q_table.json", "w") as f:
    json.dump(q_list, f)

print("\n✅ Training complete! Q-table saved to q_table.json")
print(f"   Final avg reward (last 1000 eps): {np.mean(rewards_history[-1000:]):+.2f}")

# ─── Quick evaluation ─────────────────────────────────────────────────────────
env = gym.make("Taxi-v3")
eval_rewards = []
for _ in range(100):
    state, _ = env.reset()
    total, done = 0, False
    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    eval_rewards.append(total)
env.close()

print(f"   Evaluation (100 eps): avg={np.mean(eval_rewards):+.2f}, "
      f"min={min(eval_rewards):+.2f}, max={max(eval_rewards):+.2f}")
