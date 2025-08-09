# policy.py
# NumPy MLP policy with two hidden layers and manual backprop for REINFORCE.

import numpy as np

class MLPPolicy:
    def __init__(
        self,
        state_dim: int,
        hidden1: int = 128,
        hidden2: int = 32,
        n_actions: int = 3,
        seed: int = 0
    ):
        rng = np.random.default_rng(seed)
        # He init for ReLU layers
        self.W1 = rng.normal(0, np.sqrt(2 / state_dim), size=(state_dim, hidden1))
        self.b1 = np.zeros((hidden1,), dtype=np.float32)

        self.W2 = rng.normal(0, np.sqrt(2 / hidden1), size=(hidden1, hidden2))
        self.b2 = np.zeros((hidden2,), dtype=np.float32)

        self.W3 = rng.normal(0, np.sqrt(2 / hidden2), size=(hidden2, n_actions))
        self.b3 = np.zeros((n_actions,), dtype=np.float32)

    def forward(self, s):
        """
        Forward pass; returns action probabilities and a cache for backprop.
        s: shape [S] (state vector)
        """
        z1 = s @ self.W1 + self.b1
        h1 = np.maximum(0, z1)  # ReLU

        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, z2)  # ReLU

        logits = h2 @ self.W3 + self.b3

        # softmax with stability
        m = np.max(logits)
        exp = np.exp(logits - m)
        probs = exp / (np.sum(exp) + 1e-8)

        cache = (s, z1, h1, z2, h2, logits, probs)
        return probs, cache

    def sample_action(self, s, rng: np.random.Generator):
        probs, cache = self.forward(s)
        a = rng.choice(len(probs), p=probs)
        return int(a), probs, cache

    def backprop_from_dlogits(self, cache, dlogits):
        """
        Backprop given gradient w.r.t. logits (shape [A]).
        Returns grads for ALL parameters {W1,b1,W2,b2,W3,b3}.
        """
        s, z1, h1, z2, h2, logits, probs = cache

        # Top layer
        dW3 = np.outer(h2, dlogits)   # [H2, A]
        db3 = dlogits                 # [A]
        dh2 = dlogits @ self.W3.T     # [H2]

        # Through ReLU at layer 2
        dz2 = dh2 * (z2 > 0)          # [H2]
        dW2 = np.outer(h1, dz2)       # [H1, H2]
        db2 = dz2                     # [H2]
        dh1 = dz2 @ self.W2.T         # [H1]

        # Through ReLU at layer 1
        dz1 = dh1 * (z1 > 0)          # [H1]
        dW1 = np.outer(s, dz1)        # [S, H1]
        db1 = dz1                     # [H1]

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    # kept for compatibility if something still calls grad_logp(a)
    def grad_logp(self, cache, a):
        _, _, _, _, _, _, probs = cache
        y = np.zeros_like(probs); y[a] = 1.0
        dlogits = (probs - y)  # ∇(-log π(a|s)) = probs - one_hot(a)
        return self.backprop_from_dlogits(cache, dlogits)

    def apply_grads(self, grads, lr: float):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]
        self.W3 -= lr * grads["W3"]
        self.b3 -= lr * grads["b3"]
