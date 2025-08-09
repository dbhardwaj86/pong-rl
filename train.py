# train.py
# Training harness for state-based Pong with REINFORCE (NumPy only).

import os, argparse
import numpy as np
from utils import tee_stdout, timestamp_for_filename, save_checkpoint, load_checkpoint
from env_pong import PongEnv, PongConfig
from policy import MLPPolicy
from reinforce import compute_returns, normalize, policy_gradient_step
from utils import set_seed, EMA

def run_episode(env: PongEnv, policy: MLPPolicy, rng: np.random.Generator, max_steps=2000):
    s = env.reset()
    states, actions, rewards = [], [], []
    total_r = 0.0
    for _ in range(max_steps):
        a, probs, _ = policy.sample_action(s, rng)
        s2, r, done, _ = env.step(a - 1)  # {0,1,2} -> {-1,0,1}
        states.append(s.copy())
        actions.append(a)
        rewards.append(r)
        total_r += r
        s = s2
        if done:
            break
    return np.array(states), np.array(actions), np.array(rewards), float(total_r)

def main():
    seed = 123
    set_seed(seed)
    rng = np.random.default_rng(seed)

    env = PongEnv(PongConfig(rng_seed=seed))
    policy = MLPPolicy(state_dim=6, hidden=64, n_actions=3, seed=seed)

    ema_return = EMA(alpha=0.1)

    episodes = 10000
    batch_episodes = 10        # <-- batch size for updates
    lr = 3e-3
    gamma = 0.99               # slightly lower to reduce return variance
    beta = 0.01                # entropy in LOSS (regularization strength)

    buffer_S, buffer_A, buffer_G = [], [], []
    best_score = -1e9

    for ep in range(1, episodes + 1):
        states, actions, rewards, total_r = run_episode(env, policy, rng)
        G = compute_returns(rewards, gamma=gamma)

        buffer_S.append(states)
        buffer_A.append(actions)
        buffer_G.append(G)

        avg_ret = ema_return.update(total_r)
        if total_r > best_score:
            best_score = total_r

        # Log every 10 episodes
        if ep % 10 == 0:
            print(f"Ep {ep:4d} | EpLen {len(rewards):4d} | Return {total_r:+7.3f} | EMA {avg_ret:+7.3f} | Best {best_score:+7.3f}")

        # Do an update every 'batch_episodes' episodes
        if ep % batch_episodes == 0:
            S = np.concatenate(buffer_S, axis=0)
            A = np.concatenate(buffer_A, axis=0)
            Gcat = np.concatenate(buffer_G, axis=0)
            Ahat = normalize(Gcat)        # normalize across the entire batch

            policy_gradient_step(policy, S, A, Ahat, lr=lr, beta=beta)

            buffer_S.clear(); buffer_A.clear(); buffer_G.clear()

    print("Training complete.")

if __name__ == "__main__":
    main()
