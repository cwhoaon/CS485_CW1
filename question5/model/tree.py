import numpy as np
from .node import Node
from scipy.stats import mode

class Tree():
    def __init__(self, depth, split_trial, split_mode):
        self.depth = depth
        self.split_trial = split_trial
        self.split_mode = split_mode
        self.nodes = [None for _ in range(2**self.depth - 1)]

    def make_root(self, X, y) -> Node:
        root = Node(X, y, self.split_trial, self.split_mode)
        self.nodes[0] = root
        return root

    def train(self, X, y):
        self.make_root(X, y)

        for i in range(2**(self.depth-1) - 1):
            if self.nodes[i] == None: continue

            l_node, r_node = self.nodes[i].split_node()
            self.nodes[2*i + 1] = l_node
            self.nodes[2*i + 2] = r_node
        
        return self
            
    def predict(self, X):
        # X: (batch_size, num_features)
        batch_size = X.shape[0]
        
        idx = np.arange(0, batch_size).reshape(batch_size, 1)
        X_idx = np.concatenate((X, idx), axis=1)
        predict = np.zeros(batch_size)
        X_per_nodes = [None for _ in range(len(self.nodes))]
        X_per_nodes[0] = X_idx

        for i in range(2**self.depth - 1):
            node = self.nodes[i]
            cur_X = X_per_nodes[i]

            if node==None:
                continue
            elif node.is_leaf:
                label = mode(node.y)[0]
                cur_idx = cur_X[:, -1:].flatten().astype(int)
                predict[cur_idx] = label
            else:
                X_l, X_r = node.predict(cur_X)
                X_per_nodes[2*i+1] = X_l
                X_per_nodes[2*i+2] = X_r

        return predict
