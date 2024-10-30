import random
import numpy as np

class Node():
    def __init__(self, X, y, split_trial, weak_learner, randomness, eps=0.1):
        self.X = X
        self.y = y
        self.split_trial = split_trial
        self.weak_learner = weak_learner
        self.randomness = randomness
        self.eps = eps

        self.is_leaf = True
        self.threshold = None
        self.dim = None
    
    def split_node(self):
        # node contains only one class
        if np.unique(self.y).shape[0] == 1:
            return None, None

        self.is_leaf = False
        X, y = self.X, self.y

        best_IG = -1 # best ideal IG = H, worst ideal IG = 0
        for i in range(self.split_trial):
            
            while True:
                # if d_min == d_max, choose another dim
                dim = random.randint(0, self.X.shape[1]-1)
                d_min = X[:, dim].min()
                d_max = X[:, dim].max()
                if d_min != d_max:
                    break

            rand = random.random()
            t = d_min + rand * (d_max-d_min)
            
            L_X, R_X = X[X[:, dim] < t], X[X[:, dim] >= t]
            L_y, R_y = y[X[:, dim] < t], y[X[:, dim] >= t]
        
            H, HL, HR = Node.get_entropy(y), Node.get_entropy(L_y), Node.get_entropy(R_y)
            IG = H - (L_y.shape[0] * HL + R_y.shape[0] * HR) / y.shape[0]

            if L_y.shape[0] > 0 and R_y.shape[0] > 0:
                if IG > best_IG:
                    best_IG = IG
                    self.threshold = t
                    self.dim = dim

        L_X, R_X = X[X[:, self.dim] < self.threshold], X[X[:, self.dim] >= self.threshold]
        L_y, R_y = y[X[:, self.dim] < self.threshold], y[X[:, self.dim] >= self.threshold]
        return Node(L_X, L_y, self.split_trial, self.weak_learner, self.randomness), Node(R_X, R_y, self.split_trial, self.weak_learner, self.randomness)

    def test(self, X):
        if self.is_leaf:
            print("leaf but test")
            return
        
        X_l = X[X[:,self.dim] < self.threshold]
        X_r = X[X[:,self.dim] >= self.threshold]
        return X_l, X_r
    
    @staticmethod
    def get_entropy(idxs):
        tot = idxs.shape[0]
        _, counts = np.unique(idxs, return_counts=True)

        probs = counts / tot
        entropies = -probs * np.log(probs)
        H = entropies.sum()

        return H