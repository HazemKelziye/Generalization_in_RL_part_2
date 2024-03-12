import numpy as np


def generate_dataset():
    pass

def discretize_actions(action, bins_num=6, min_val=-2.0, max_val=2.0):
    """Discretize action space"""
    intervals = np.linspace(min_val, max_val, bins_num)

    # digitize, returns the index of action in interval
    return intervals[np.digitize(action, intervals) - 1]
