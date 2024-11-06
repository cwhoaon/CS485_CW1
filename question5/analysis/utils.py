import numpy as np



def get_avg_leaf_entropy(model):
    Hs = []
    for tree in model.trees:
        for node in tree.nodes:
            if node is not None and node.is_leaf:
                H = get_entropy(node.y)
                Hs.append(H)
    return sum(Hs) / len(Hs)

def get_data_ratio(tree):
    pass

def get_entropy(labels):
    tot = labels.shape[0]
    _, counts = np.unique(labels, return_counts=True)

    probs = counts / tot
    entropies = -probs * np.log(probs)
    H = entropies.sum()

    return H