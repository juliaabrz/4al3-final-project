##############################
#    K Nearest Neighbours    #
##############################
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import KFold
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocessing as pp

class KNearestNeighbours:
    def __init__(self, k, input, target):
        # get data from preprocessing
        self.k = k
        # drop sensitive data 
        drop_cols = ['Education','Income','Sex','MentHlth']
        input = input.drop(columns = drop_cols)
        # determine the number of features (-1) for later on
        self.num_cols = 20 - len(drop_cols)
        self.input = np.array(input)
        self.target = np.array(target)
        self.target = self.target.reshape(-1,1)

    def euclidean_distance (self, x1, x2): # calculates the euclidean distance
        return np.sqrt(np.sum(np.square(x1-x2)))
    
    def find_distances (self,x):
        
        distances = []
        # find the distance from each sample with the current data point
        for sample in self.data:
            distance = self.euclidean_distance(sample[:self.num_cols],x)
            distances.append((distance,sample[self.num_cols])) # add distance and class to list
        
        return distances

    def classify (self,x):
        # get the distances of all data points and their class 
        distances = self.find_distances(x)
        # sort based on distance
        distances.sort(key= lambda x: x[0])

        # get majority of first k classes 
        classes = []
        for i in range (self.k):
            classes.append(distances[i][1])

        # get count of 1s and 0s
        ones = classes.count(1.0)
        zeros = classes.count(0.0)

        if ones > zeros:
            return 1.0
        else:
            return 0.0

    def k_fold_validation (self,k_value):

        accuracies = []
        recalls = []
        precisions = []
        f1_scores = []
        # use KFold from sklearn to split the data 
        k_fold_split = KFold(n_splits=k_value,shuffle=True, random_state = 42)

        # go through each fold
        for train_index, test_index in k_fold_split.split(self.input,self.target):
            #get the train and test sets from the input and target variables
            train_x = self.input[train_index]
            train_y = self.target[train_index]
            test_x = self.input[test_index]
            test_y = self.target[test_index]


            #concatenate together x and y, last label is classification
            self.data = np.hstack((train_x,train_y))
            
            y_pred = np.empty(test_x.shape[0])
            for d in range (test_x.shape[0]):
                y_pred[d] = self.classify(test_x[d])
            
            # evaluate model for each fold
            accuracy = accuracy_score(test_y, y_pred)
            print ("Model accuracy:",accuracy)

            recall = recall_score(test_y,y_pred)
            print ("Model recall:",recall)

            precision = precision_score(test_y,y_pred)
            print ("Model precision", precision)

            f1 = f1_score(test_y,y_pred)
            print ("F1 score:",f1)

            accuracies.append(accuracy)
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1)
        

        # convert arrays to np arrays for easy calculation of mean scores
        accuracies = np.array(accuracies)
        average_acc = np.mean(accuracies,axis=0)
        recalls = np.array(recalls)
        average_rec = np.mean(recalls,axis=0)
        f1_scores = np.array(f1_scores)
        average_f1 = np.mean(f1_scores,axis=0)
        precisions = np.array(precisions)
        average_pre = np.mean(precisions, axis=0)

        return (average_acc,average_rec,average_pre,average_f1)

def run_knn():

    x_train, y_train = pp.preprocessing(0.02, True) # get data from preprocessing function, will be doing k fold
    print("Shape of data used", x_train.shape)


    model = KNearestNeighbours(5,x_train, y_train)
    # classify data
    acc, rec, prec, f1 = model.k_fold_validation(5)
    print ("Average model accuracy:",acc)
    print ("Average model recall:",rec)
    print ("Average model precision", prec)
    print ("Average F1 score:",f1)

run_knn()
