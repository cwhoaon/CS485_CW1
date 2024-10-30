import numpy as np
from node import Node

class Tree():
    def __init__(self, params):
        self.depth
        self.nodes = [None for _ in range(2**self.depth - 1)]

    def grow_tree(self, X, y):
        self.make_root()
        for i in range(2**(self.depth-1) - 1):
            l_node, r_node = self.nodes[i].split_node(X, y)
            self.nodes[2**i + 1] = l_node
            self.nodes[2**i + 2] = r_node



    def make_root(self):
        root = Node(self.param)
        self.nodes[0] = root