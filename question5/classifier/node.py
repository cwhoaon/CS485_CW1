import random
from util import get_entropy

class Node():
    def __init__(self):
        self.is_leaf
        self.idx = []
        self.dim
        self.threshold
        self.feature_num
        self.param
    
    def split_node(self, X, y):
        self.idx = y

        best_IG = -1 # best ideal IG = H, worst ideal IG = 0
        for i in range(self.param['num_split_trials']):
            dim = random.randrange(self.featurn_num)
            d_min = X[:, dim].min()
            d_max = X[:, dim].max()

            t = d_min + random.random() * (d_max-d_min)

            L_idx = y[X[:, dim] < t]
            R_idx = y[(X[:, dim] < t).logical_not()]
        
            H, HL, HR = get_entropy(self.idx), get_entropy(L_idx), get_entropy(R_idx)
            IG = H - (L_idx.shape[0] * HL + R_idx.shape[0] * HR) / self.idx.shape[0]

            if L_idx.shape[0] > 0 and R_idx.shape[0] > 0:
                if IG > best_IG:
                    best_IG = IG
                    self.threshold = t

        return 




            


