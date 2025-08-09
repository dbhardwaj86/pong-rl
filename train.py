# train.py
# Training harness for state-based Pong with REINFORCE (NumPy only).

import os
import argparse
import numpy as np

from utils import (
    tee_stdout,
    set_seed,
    EMA,
    save_checkpoint,
    load_checkpoint,
)
from env_pong import PongEnv, PongConfig
from policy import MLPPolicy
from reinforce import compute_returns, normalize, policy_gradient_step


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--batch", type=int, default=30, help="episodes per policy update")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--gamma", type=float, default=0.997)

    # Entropy regularization schedule
    p.add_argument("--beta_start", type=float, default=0.02)
    p.add_argument("--beta_min", type=float, default=0.002)
    p.add_argument("--beta_decay", type=float, default=0.995)

    # Checkpointing / logging
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume")
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--ckpt_path", type=str, default="pong_ckpt.pkl")
    p.add_argument("--logdir", type=str, default="logs")

    # Reproducibility
    p.add_argument("--seed", type=int, default=123)

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, "train.log")
    tee_stdout(log_path)
    print(f"Logging to {log_path}")
    print(f"Args: {vars(args)}")

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = PongEnv(PongConfig(rng_seed=args.seed))
    policy = policy = MLPPolicy(state_dim=6, hidden1=128, hidden2=32, n_actions=3, seed=args.seed)

    start_ep = 1
    ema_return = EMA(alpha=0.1)

    # Resume support (expects load_checkpoint to return (obj, meta))
    if args.resume:
        try:
            obj, meta = load_checkpoint(args.resume)
            if "policy" in obj:
                # Load weights
                p_state = obj["policy"]
                # Assign if shapes match
                policy.W1 = p_state["W1"]
                policy.b1 = p_state["b1"]
                policy.W2 = p_state["W2"]
                policy.b2 = p_state["b2"]
            if "rng_state" in obj:
                rng = np.random.default_rng()
                rng.bit_generator.state = obj["rng_state"]
            if meta and "next_episode" in meta:
                start_ep = int(meta["next_episode"])
            if meta and "ema" in meta:
                ema_return.value = float(meta["ema"])
            print(f"Resumed from {args.resume} at episode {start_ep}.")
        except Exception as e:
            print(f"WARNING: Failed to resume from {args.resume}: {e}")

    episodes = args.episodes
    batch_episodes = args.batch
    lr = args.lr
    gamma = args.gamma

    # Entropy schedule
    beta = args.beta_start
    beta_min = args.beta_min
    beta_decay = args.beta_decay

    buffer_S, buffer_A, buffer_G = [], [], []
    best_score = -1e9

    for ep in range(start_ep, episodes + 1):
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
            print(
                f"Ep {ep:4d} | EpLen {len(rewards):4d} | Return {total_r:+7.3f} | "
                f"EMA {avg_ret:+7.3f} | Best {best_score:+7.3f}"
            )

        # Update after a batch of episodes
        if ep % batch_episodes == 0:
            S = np.concatenate(buffer_S, axis=0)
            A = np.concatenate(buffer_A, axis=0)
            Gcat = np.concatenate(buffer_G, axis=0)
            Ahat = normalize(Gcat)  # normalize across the entire batch

            policy_gradient_step(policy, S, A, Ahat, lr=lr, beta=beta)

            buffer_S.clear(); buffer_A.clear(); buffer_G.clear()

            # Anneal entropy beta
            beta = max(beta_min, beta * beta_decay)

        # Periodic checkpoint
        if ep % args.ckpt_every == 0:
            to_save = {
                "policy": {
                    "W1": policy.W1, "b1": policy.b1,
                    "W2": policy.W2, "b2": policy.b2
                },
                "rng_state": rng.bit_generator.state,
            }
            meta = {
                "next_episode": ep + 1,
                "ema": ema_return.value,
                "beta": beta,
                "gamma": gamma,
                "lr": lr,
                "batch": batch_episodes,
                "seed": args.seed,
            }
            # NOTE: positional args per your utils.py: (obj, meta, path)
            save_checkpoint(to_save, meta, args.ckpt_path)
            print(f"Checkpoint saved at {args.ckpt_path} (episode {ep})")

    print("Training complete.")


if __name__ == "__main__":
    main()
