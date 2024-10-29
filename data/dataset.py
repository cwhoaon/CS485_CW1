import numpy as np
from scipy import io

PATH = "./data/face.mat"

class FaceData():
    def __init__(self):
        mat_file = io.loadmat(PATH)
        self.data = mat_file['X'].T
        self.target = mat_file['l'].squeeze()
