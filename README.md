# Pong RL – Sprint 0 & 1 (From-Scratch, NumPy Only)

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

## What's new in this patch
- **Batched updates:** accumulate 10 episodes, then apply one gradient step (lower variance).
- **Entropy in loss:** proper entropy regularization (`beta=0.01`) rather than reward-side shaping.
- **No reward-side entropy:** cleaner credit assignment.
- **Gamma 0.99 & max_steps 2000:** more stable returns without truncation artifacts.

# Pong RL – Phase A (Sprint 0–1)

Minimal-from-scratch RL agent in NumPy that learns to play a simplified **state-based Pong**.

---

## Files
- `env_pong.py` – Environment with easier opponent, biased serve, shaping rewards.
- `policy.py` – MLP policy with manual backprop. Now supports `save()`/`load()`.
- `reinforce.py` – Return computation, advantage normalization, gradient clipping.
- `train.py` – Training harness with entropy bonus, tuned hyperparameters, checkpoint/resume, and logging to both console and file.
- `utils.py` – Seed control, EMA tracker.
- `requirements.txt` – NumPy only.
- `README.md` – Instructions.

---

## Key training tweaks
- Nerfed left paddle speed, bigger paddle hitbox.
- Serve aimed at agent to guarantee early contact.
- Dense shaping reward for paddle-ball alignment.
- Entropy bonus to maintain exploration.
- Longer episodes, reduced LR, increased gamma.
- Gradient clipping for stability.

---

## Running Training

Start fresh training:
```bash
python train.py
