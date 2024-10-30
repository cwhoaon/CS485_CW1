import numpy as np

def get_entropy(idxs):
    tot = idxs.shape[0]
    _, counts = np.unique(idxs, return_counts=True)

    probs = counts / tot
    entropies = -probs * np.log(probs)
    H = entropies.sum()

    return H