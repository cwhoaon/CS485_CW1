import numpy as np
from .tree import Tree
from scipy.stats import mode

class RandomForest:
    def __init__(self, num_trees, depth, split_trial, weak_learner, randomness):
        self.num_trees = num_trees
        self.depth = depth
        self.split_trial = split_trial
        self.weak_learner = weak_learner
        self.randomness = randomness

        self.trees = []
    
    def train(self, X, y):
        self.trees = [Tree(self.depth, self.split_trial, self.weak_learner, self.randomness) for _ in range(self.num_trees)]
        for i, t in enumerate(self.trees):
            print(f"{i+1}th tree training")
            X, y = RandomForest.bagging(X, y)
            t.train(X, y)
    
    def predict(self, X):
        vote_list = []
        for i, tree in enumerate(self.trees):
            predict = tree.predict(X)
            vote_list.append(predict)
        
        vote = np.stack(vote_list)
        output = mode(vote, axis=0)[0].astype(int)

        return output

    @staticmethod
    def bagging(X, y):
        num_sample = X.shape[0]
        idx = np.random.randint(0, num_sample, size=num_sample)
        X, y = X[idx], y[idx]
        return X, y