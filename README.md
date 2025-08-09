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
