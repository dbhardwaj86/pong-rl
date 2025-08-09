# policy.py
# NumPy MLP policy with manual backprop for policy-gradient (REINFORCE).

import numpy as np

class MLPPolicy:
    def __init__(self, state_dim: int, hidden: int, n_actions: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He init for ReLU
        self.W1 = rng.normal(0, np.sqrt(2/state_dim), size=(state_dim, hidden))
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2/hidden), size=(hidden, n_actions))
        self.b2 = np.zeros((n_actions,), dtype=np.float32)

    def forward(self, s):
        """Forward pass; returns probs and cached intermediates for backprop."""
        z1 = s @ self.W1 + self.b1
        h = np.maximum(0, z1)  # ReLU
        logits = h @ self.W2 + self.b2
        # softmax
        logits = logits - np.max(logits)  # numerical stability
        exp = np.exp(logits)
        probs = exp / (np.sum(exp) + 1e-8)
        cache = (s, z1, h, logits, probs)
        return probs, cache

    def sample_action(self, s, rng: np.random.Generator):
        probs, cache = self.forward(s)
        a = rng.choice(len(probs), p=probs)
        return int(a), probs, cache

    # New: generic backprop from dlogits (lets us combine multiple loss terms)
    def backprop_from_dlogits(self, cache, dlogits):
        s, z1, h, logits, probs = cache
        dW2 = np.outer(h, dlogits)         # [H, A]
        db2 = dlogits                      # [A]
        dh = dlogits @ self.W2.T           # [H]
        dz1 = dh * (z1 > 0)                # ReLU
        dW1 = np.outer(s, dz1)             # [S, H]
        db1 = dz1
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # (kept for compatibility if you still call it elsewhere)
    def grad_logp(self, cache, a):
        s, z1, h, logits, probs = cache
        y = np.zeros_like(probs); y[a] = 1.0
        dlogits = (probs - y)
        return self.backprop_from_dlogits(cache, dlogits)

    def apply_grads(self, grads, lr: float):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]
