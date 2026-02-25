# 🚕 Taxi-v3 Reinforcement Learning Agent

A Q-Learning agent trained on OpenAI Gymnasium's Taxi-v3 environment.
Includes a fully playable browser-based game UI where you can watch 
the trained agent navigate, or take control yourself.

---

## 🧠 What is this?

The Taxi-v3 environment is a 5×5 grid where a taxi must:
1. Navigate to a passenger's location
2. Pick them up
3. Drop them off at the correct destination

The agent learns this entirely through trial and error using 
**Q-Learning** — a model-free reinforcement learning algorithm.
After 10,000 episodes it consistently solves the environment 
in under 15 steps with a positive reward.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `taxi_rl.ipynb` | Full training notebook — run on Google Colab |
| `taxi_rl_train.py` | Standalone Python training script |
| `taxi_rl_game.html` | Browser game UI — open locally or via GitHub Pages |

---

## 🚀 How to Run

### Train the Agent (Google Colab — no install needed)
1. Open `taxi_rl.ipynb` in [Google Colab](https://colab.research.google.com)
2. Click **Runtime → Run All**
3. Training completes in ~2 minutes
4. Download the generated `q_table.json`

### Play the Game
1. Open `taxi_rl_game.html` in any browser
2. The game runs immediately with a built-in agent
3. To use your trained agent: click the file picker and load `q_table.json`
4. Switch between **Agent mode** (watch it play) and **Manual mode** (play yourself)

---

## 🎮 Game Controls

| Key | Action |
|-----|--------|
| ↑ ↓ ← → | Move taxi |
| P | Pick up passenger |
| D | Drop off passenger |

---

## 📊 Algorithm

**Q-Learning** with ε-greedy exploration.

| Parameter | Value |
|-----------|-------|
| Episodes | 10,000 |
| Learning rate (α) | 0.10 |
| Discount factor (γ) | 0.99 |
| Epsilon start | 1.0 |
| Epsilon end | 0.01 |
| Epsilon decay | 0.0005 |

The Q-table has **500 states × 6 actions** = 3,000 values,
updated each step using the Bellman equation:
```
Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s',a') − Q(s,a) ]
```

---

## 📈 Results

After training:
- ✅ Average reward: **+8 to +9** (random agent scores ~−200)
- ✅ Average steps to solve: **~13**
- ✅ Success rate: **~98%** over 500 evaluation episodes

---

## 🛠 Tech Stack

- Python 3.10+
- [Gymnasium](https://gymnasium.farama.org/) — environment
- NumPy — Q-table operations
- Matplotlib — training curves
- Vanilla HTML / CSS / JS — game UI (zero dependencies)

---

## 📌 What I Learned

- How Q-Learning and the Bellman equation work in practice
- The exploration vs exploitation tradeoff (ε-greedy)
- How to decode Gymnasium's state encoding
- Building an RL environment from scratch in JavaScript for the UI

---

## 🔮 Future Improvements

- [ ] Replace Q-table with a Deep Q-Network (DQN) using PyTorch
- [ ] Add live reward and epsilon decay graphs in the UI
- [ ] Visualize the Q-table as a heatmap
- [ ] Try harder environments (LunarLander, CartPole)
