import numpy as np
from .tree import Tree
from scipy.stats import mode
from joblib import Parallel, delayed

class RandomForest:
    def __init__(self, num_trees, depth, split_trial, split_mode, n_jobs=1):
        self.num_trees = num_trees
        self.depth = depth
        self.split_trial = split_trial
        self.split_mode = split_mode
        self.n_jobs=n_jobs

        self.trees = []
    
    def train(self, X, y):
        self.trees = [Tree(self.depth, self.split_trial, self.split_mode) for _ in range(self.num_trees)]

        def train_tree(i, tree, X, y):
            # print(f"Tree{i+1} training")
            X, y = RandomForest.bootstrap(X, y)
            return tree.train(X, y)

        self.trees = Parallel(n_jobs=self.n_jobs)(delayed(train_tree)(i, t, X, y) for i, t in enumerate(self.trees))
            
    def predict(self, X):
        vote_list = Parallel(n_jobs=self.n_jobs)(delayed(t.predict)(X) for t in self.trees)
        vote = np.stack(vote_list)
        output = mode(vote, axis=0)[0].astype(int)

        return output

    @staticmethod
    def bootstrap(X, y):
        num_sample = X.shape[0]
        idx = np.random.randint(0, num_sample, size=num_sample)
        X, y = X[idx], y[idx]
        return X, y