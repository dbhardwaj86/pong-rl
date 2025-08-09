# reinforce.py
# Returns, advantage normalization, and a single policy-gradient update step.

import numpy as np
from policy import MLPPolicy

def compute_returns(rewards, gamma=0.99):
    G = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G

def normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + eps)

def policy_gradient_step(policy: MLPPolicy, states, actions, advantages, lr=1e-3, beta=0.01):
    """
    One update using REINFORCE with entropy regularization in the LOSS:
    L = -E[adv * log pi(a|s)] + beta * sum_i p_i log p_i
    (i.e., -H bonus; smaller beta => weaker regularization)
    """
    agg = {"W1": 0, "b1": 0, "W2": 0, "b2": 0}
    for s, a, adv in zip(states, actions, advantages):
        probs, cache = policy.forward(s)

        # Policy term: adv * (probs - onehot(a))
        y = np.zeros_like(probs); y[a] = 1.0
        dlogits = (probs - y) * adv

        # Entropy term gradient: d/dlogits [sum_i p_i log p_i]
        # = probs * (log probs + 1 - <log probs + 1>_p)
        t = np.log(probs + 1e-8) + 1.0
        t_bar = np.sum(probs * t)
        entropy_grad_logits = probs * (t - t_bar)

        dlogits = dlogits + beta * entropy_grad_logits

        grads = policy.backprop_from_dlogits(cache, dlogits)
        for k in agg:
            agg[k] = agg[k] + grads[k]

    # Average over batch
    for k in agg:
        agg[k] /= max(1, len(states))
        agg[k] = np.clip(agg[k], -1.0, 1.0)  # gradient clipping

    policy.apply_grads(agg, lr)
