# utils.py
import numpy as np
import random
import os, sys, pickle
from contextlib import contextmanager
from datetime import datetime


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class _TeeStdout:
    def __init__(self, filepath, mode="a"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # line-buffered text for immediate writes on Windows too
        self._f = open(filepath, mode, buffering=1, encoding="utf-8")
        self._console = sys.__stdout__

    def write(self, data):
        self._console.write(data)
        self._f.write(data)

    def flush(self):
        self._console.flush()
        self._f.flush()

    def close(self):
        try: self._f.close()
        except: pass

@contextmanager
def tee_stdout(logfile_path: str):
    """
    Send *all* prints to the terminal *and* a log file, without changing print() calls.
    Usage:
        with tee_stdout("logs/run-2025-08-09_12-34-56.txt"):
            ... your normal training loop (print as usual) ...
    """
    tee = _TeeStdout(logfile_path)
    old = sys.stdout
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = old
        tee.close()

def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_checkpoint(path: str, policy_obj, meta: dict):
    """
    Save the full policy object + training meta using pickle (stdlib).
    Keeps this repo numpy-only (no extra deps).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"policy": policy_obj, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_checkpoint(path: str):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return blob["policy"], blob.get("meta", {})
    
class EMA:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None
    def update(self, x: float):
        if self.value is None:
            self.value = x
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * x
        return self.value
