# utils.py
import os
import sys
import pickle
import random
import numpy as np
from datetime import datetime

class EMA:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def timestamp_for_filename():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def tee_stdout(log_path):
    """
    Redirects stdout to both terminal and a log file.
    Returns a file-like object that should be closed at the end.
    """
    log_file = open(log_path, "w", buffering=1)

    class Tee:
        def __init__(self, files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
        def flush(self):
            for f in self.files:
                f.flush()
        def close(self):
            for f in self.files:
                f.close()

    tee = Tee([sys.__stdout__, log_file])
    sys.stdout = tee
    return tee


def save_checkpoint(*args):
    """
    Flexible save:
      save_checkpoint(data, path)
      save_checkpoint(data, meta, path)
    """
    if len(args) == 2:
        data, path = args
        meta = {}
    elif len(args) == 3:
        data, meta, path = args
    else:
        raise TypeError("save_checkpoint expects (data, path) or (data, meta, path)")

    payload = {
        "data": data,
        "meta": meta,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path):
    """
    Loads checkpoint and returns (data, meta).
    If file does not exist, returns None.
    """
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "data" in payload:
        return payload["data"], payload.get("meta", {})
    else:
        # Legacy: assume the whole payload is `data`, meta empty
        return payload, {}
