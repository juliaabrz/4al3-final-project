##############################
#    K Nearest Neighbours    #
##############################
import pandas as pd
import numpy as np
import sklearn
import sys
import os


sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../../'))
import preprocessing as pp

print (sys.path)

class KNearestNeighbours:
    def __init__(self, k):
        # get data from preprocessing
        self.k = k
    
    def get_data(self):
        x_train, x_test, y_train, y_test = pp.preprocessing()
        print (x_train.shape)


model = KNearestNeighbours(5)
model.get_data()