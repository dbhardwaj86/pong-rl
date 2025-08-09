# reinforce.py
# Utilities for REINFORCE with NumPy policies.

import numpy as np

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns G_t for a single episode reward array."""
    G = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G

def normalize(x, eps=1e-8):
    """Zero-mean, unit-std normalization (safe for small variance)."""
    m = np.mean(x)
    s = np.std(x)
    return (x - m) / (s + eps)

def policy_gradient_step(policy, S, A, Adv, lr=3e-3, beta=0.01):
    """
    One PG step over a batch of (S, A, Adv).
    - policy: has forward(), backprop_from_dlogits(), apply_grads()
    - Entropy bonus: use a KL-to-uniform surrogate: beta * KL(p || U)
      -> d/dlogits ≈ beta * (p - 1/K)
    """
    K = None
    agg = None

    for s, a, adv in zip(S, A, Adv):
        probs, cache = policy.forward(s)
        if K is None:
            K = len(probs)

        # REINFORCE gradient w.r.t. logits: (p - onehot(a)) * advantage
        y = np.zeros_like(probs); y[a] = 1.0
        dlogits = (probs - y) * float(adv)

        # Entropy surrogate: push toward uniform distribution
        if beta and beta > 0.0:
            dlogits += beta * (probs - (1.0 / K))

        # Backprop through the network
        grads = policy.backprop_from_dlogits(cache, dlogits)

        # Dynamically initialize aggregator with zeros matching shapes
        if agg is None:
            agg = {k: np.zeros_like(v) for k, v in grads.items()}

        # Sum grads for whatever keys the policy returns (W1..W3, b1..b3 etc.)
        for k, g in grads.items():
            agg[k] += g

    # Average over batch (good practice; lr can then be more stable)
    batch_n = max(1, len(S))
    for k in agg.keys():
        agg[k] /= batch_n

    policy.apply_grads(agg, lr)
