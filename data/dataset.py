import numpy as np
import math
from scipy import io

class FaceData():
    def __init__(self, path, test_ratio=0.2):
        mat_file = io.loadmat(path)

        self.num_class = 52
        self.data_per_class = 10
        self.num_test = math.ceil(self.data_per_class * test_ratio)
        self.num_train = self.data_per_class - self.num_test

        X = mat_file['X'].T.reshape(self.num_class, self.data_per_class, -1)
        y = mat_file['l'].squeeze().reshape(self.num_class, self.data_per_class)
        self.num_feature = X.shape[-1]

        self.train_X = X[:, :self.num_train].reshape(-1, self.num_feature)
        self.train_y = y[:, :self.num_train].flatten()
        self.test_X = X[:, self.num_train:].reshape(-1, self.num_feature)
        self.test_y = y[:, self.num_train:].flatten()
