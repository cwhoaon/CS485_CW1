import random
import numpy as np

class Node():
    def __init__(self, X, y, split_trial, split_mode):
        self.X = X
        self.y = y
        self.split_trial = split_trial
        self.split_mode = split_mode

        self.is_leaf = True
        self.weak_learner = None

        assert split_mode in [
            "axis-align",
            "two-pixel-test"
        ], f"Wrong split mode: {split_mode}"
    
    def split_node(self):
        # node contains only one class
        if np.unique(self.y).shape[0] == 1:
            return None, None

        self.is_leaf = False
        X, y = self.X, self.y

        best_IG = -1 # best ideal IG = H, worst ideal IG = 0
        best_weak_learner = None

        for i in range(self.split_trial):
            if self.split_mode == "axis-align":
                weak_learner = AxisAlign(X)
            elif self.split_mode == "two-pixel-test":
                weak_learner = TwoPixelTest(X)

            L_idx, R_idx = weak_learner.get_split_idx(X)
            L_X, R_X = X[L_idx], X[R_idx]
            L_y, R_y = y[L_idx], y[R_idx]
        
            H, HL, HR = Node.get_entropy(y), Node.get_entropy(L_y), Node.get_entropy(R_y)
            IG = H - (L_y.shape[0] * HL + R_y.shape[0] * HR) / y.shape[0]
            
            if L_y.shape[0] > 0 and R_y.shape[0] > 0:
                if IG > best_IG:
                    best_IG = IG
                    best_weak_learner = weak_learner
        
        assert best_weak_learner!=None, "No weak learner"
        
        self.weak_learner = best_weak_learner

        L_idx, R_idx = self.weak_learner.get_split_idx(X)
        L_X, R_X = X[L_idx], X[R_idx]
        L_y, R_y = y[L_idx], y[R_idx]

        return (
            Node(L_X, L_y, self.split_trial, self.split_mode), 
            Node(R_X, R_y, self.split_trial, self.split_mode)
        )

    def predict(self, X):
        if self.is_leaf:
            print("leaf but test")
            return

        L_idx, R_idx = self.weak_learner.get_split_idx(X)
        L_X, R_X = X[L_idx], X[R_idx]

        return L_X, R_X
    
    @staticmethod
    def get_entropy(labels):
        tot = labels.shape[0]
        _, counts = np.unique(labels, return_counts=True)

        probs = counts / tot
        entropies = -probs * np.log(probs)
        H = entropies.sum()

        return H


### Implemetation of various weak-learners

class AxisAlign():
    def __init__(self, X):
        dim, t = self.pick_criteria(X)
        self.dim = dim
        self.threshold = t

    def pick_criteria(self, X):
        while True:
            # if d_min == d_max, choose another dim
            dim = random.randint(0, X.shape[1]-1)
            d_min = X[:, dim].min()
            d_max = X[:, dim].max()
            if d_min != d_max:
                break
        t = d_min + random.random() * (d_max-d_min)

        return dim, t
        
    def get_split_idx(self, X):
        L_idx = X[:,self.dim] <= self.threshold
        R_idx = X[:,self.dim] > self.threshold
        return L_idx, R_idx
    
class TwoPixelTest():
    def __init__(self, X):
        dims, t = self.pick_criteria(X)
        self.dims = dims
        self.threshold = t

    def pick_criteria(self, X):
        D = X.shape[1]

        while True:
            dims = np.random.choice(np.arange(0, D), size=2, replace=False)
            X_dims = X[:, dims].astype(int)
            X_tpt = X_dims[:, 0] - X_dims[:, 1]
            d_min = X_tpt.min()
            d_max = X_tpt.max()
            if d_min != d_max:
                break
        t = d_min + random.random() * (d_max-d_min)

        return dims, t

    def get_split_idx(self, X):
        X = X[:, self.dims].astype(int)
        X = X[:, 0] - X[:, 1]

        L_idx = X <= self.threshold
        R_idx = X > self.threshold

        return L_idx, R_idx


class AxisAlignFast():
    def __init__(self, X, split_trial):
        dim, t = self.pick_criteria(X, split_trial)
        self.dim = dim
        self.threshold = t
        self.split_trial = split_trial

    def pick_criteria(self, X, split_trial):
        mask = X.min(axis=1) != X.max(axis=1) #Only use True dim
        X_mask = X[mask]
        D = X_mask.shape[1]
        dim = np.random.choice(np.arange(D), split_trial)
        X_dim = X_mask[:, dim]

        t = np.random.random(split_trial)
        t = X_dim.min(axis=1) + t * (X_dim.max(axis=1) - X_dim.min(axis=1))

        return dim, t
        
    def get_split_idx(self, X):
        L_idx = X[:,self.dim] <= self.threshold
        R_idx = X[:,self.dim] > self.threshold
        return L_idx, R_idx

def get_entropy(labels):
    
    tot = labels.shape[0]
    _, counts = np.unique(labels, return_counts=True)

    probs = counts / tot
    entropies = -probs * np.log(probs)
    H = entropies.sum()

    return H