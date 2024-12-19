##############################
#    K Nearest Neighbours    #
##############################
import pandas as pd
import numpy as np
import sklearn
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.neighbors import BallTree
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class KNearestNeighbours:
    def __init__(self, k, input, target):
        # get data from preprocessing
        self.k = k
        # drop sensitive data 
        drop_cols = ['Education','Income','Sex','MentHlth']
        input = input.drop(columns = drop_cols)
        print ("Shape of data used", input.shape)
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

    def find_distances_ball (self,x):
        ball_tree = BallTree(self.data[:,:self.num_cols])
        point = [x]
        distances, indices = ball_tree.query(point, k=self.k)
        return distances, indices
        

    def classify_ball(self,x):
        distances, indices = self.find_distances_ball(x)
        
        # get majority of first k classes
        classes = []
        for i in indices[0]:
            #print (i,self.data[i][self.num_cols])
            classes.append(self.data[i][self.num_cols])

        # get count of 1s and 0s
        ones = classes.count(1.0)
        zeros = classes.count(0.0)

        if ones > zeros:
            return 1.0
        else:
            return 0.0
        

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

    def k_fold_validation (self,k_value,ball, balance, up):

        accuracies = []
        recalls = []
        precisions = []
        f1_scores = []
        self.input,self.target = shuffle(self.input,self.target)
        # use KFold from sklearn to split the data 
        k_fold_split = StratifiedKFold(n_splits=k_value,shuffle=True, random_state = 42)

        # go through each fold
        for train_index, test_index in k_fold_split.split(self.input,self.target):
            #get the train and test sets from the input and target variables
            train_x = self.input[train_index]
            train_y = self.target[train_index]
            test_x = self.input[test_index]
            test_y = self.target[test_index]

            # balance the training data so we train on balanced classes
            if balance:
                train_x,train_y = balance_classes(train_x, train_y,up)
            #test_x,test_y = balance_classes(test_x,test_y)
            # Count the number of samples for each class
            train_distribution = pd.Series(train_y.reshape(-1)).value_counts()
            test_distribution = pd.Series(test_y.reshape(-1)).value_counts()
    
            #print(f"  Training class distribution: \n{train_distribution}")
            #print(f"  Testing class distribution: \n{test_distribution}")
            #print("-" * 50)


            #concatenate together x and y, last label is classification
            self.data = np.hstack((train_x,train_y))
            
            y_pred = np.empty(test_x.shape[0])
            for d in range (test_x.shape[0]):
                if ball:
                    y_pred[d] = self.classify_ball(test_x[d])
                else:
                    y_pred[d] = self.classify(test_x[d])
            
            # evaluate model for each fold
            accuracy = accuracy_score(test_y, y_pred)
            #print ("Model accuracy:",accuracy)

            recall = recall_score(test_y,y_pred)
            #print ("Model recall:",recall)

            precision = precision_score(test_y,y_pred)
            #print ("Model precision", precision)

            f1 = f1_score(test_y,y_pred)
            #print ("F1 score:",f1)

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

# preprocessing function is in this file for the sake of submission
def preprocessing(percentage, kfold):
    file_path='diabetes_binary_health_indicators_BRFSS2015.csv'
    # Load the dataset
    data = pd.read_csv(file_path)
    data = shuffle(data)
    data = data[:int(len(data)*percentage)] # select the number of samples you want to use
    
    # drop features that have nan
    data = data.dropna()

    # Separate target and features
    target = 'Diabetes_binary'
    X = data.drop(columns=[target, 'Stroke'])
    y = data[target]

    # # correlation analysis commented out for now since still working on it
    #correlations = X.corrwith(y)
    #selected_features = correlations[correlations.abs() > 0.1].index
    #print("Selected features based on correlation:", selected_features.tolist())
    # X = X[selected_features]

    numerical_columns = ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Income' , 'Education', 'GenHlth' ] 
    existing_num_cols = []
    
    for feature in numerical_columns:
        if feature in numerical_columns :
            existing_num_cols.append(feature)
    # Normalize numerical features using Min-Max Scaling
    scaler = MinMaxScaler()
    X[existing_num_cols] = scaler.fit_transform(X[existing_num_cols])

    #X_balanced, y_balanced = balance_classes
    if not kfold:
        # X = data.drop(columns=['Income'])
        X_balanced, y_balanced = balance_classes(X, y)

        # Split the data into training and testing sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    # if for k fold, we want to return all the data
    return X, y

def balance_classes(X, y,up):
    #print ("before balancing:",X.shape, y.shape)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    # count samples
    class_counts = y.value_counts()
    #unique_vals, counts = np.unique(y,return_counts = True)
    #class_counts = dict(zip(unique_vals,counts))

    #print (class_counts)
    max_count = max(class_counts.values)
    min_count = min(class_counts.values)
    #print (y)
        
    # X_balanced and y_balanced store the balanced classes
    X_balanced = []
    y_balanced = []
    for cls in class_counts.keys():
        #print(cls)
        X_class = X[y.squeeze() == cls]
        y_class = y[y.squeeze() == cls]
        #print(f"Shape of X: {X_class.shape}, Shape of y: {y_class.shape}")

        # upsampling 
        if class_counts[cls] < max_count:
            multiplier = max_count // class_counts[cls]
            remainder = max_count % class_counts[cls]
            X_upsampled = pd.concat([X_class] * multiplier + [X_class.sample(remainder, replace=True)])
            y_upsampled = pd.concat([y_class] * multiplier + [y_class.sample(remainder, replace=True)])
        else:
            X_upsampled = X_class
            y_upsampled = y_class
            
        # downsampling 
        if class_counts[cls] > min_count:
            # Downsample the majority class to match the minority class size
            X_downsampled = X_class.sample(min_count, replace=False)  # Sample without replacement
            y_downsampled = y_class.sample(min_count, replace=False)
        else:
            X_downsampled = X_class
            y_downsampled = y_class

        # append upsampling or downsampling depending on technqiue selected
        if up:
            X_balanced.append(X_upsampled)
            y_balanced.append(y_upsampled)
        else:
            X_balanced.append(X_downsampled)
            y_balanced.append(y_downsampled)
        
    X_balanced = pd.concat(X_balanced)
    y_balanced = pd.concat(y_balanced)
    #print ("after balancing:",X_balanced.shape,y_balanced.shape)
    return X_balanced.to_numpy(), y_balanced.to_numpy()

def run_knn_ball_tree():
    x_train, y_train = preprocessing(0.02, True) # get data from preprocessing function, will be doing k fold

    model = KNearestNeighbours(33,x_train, y_train)
    # classify data
    acc, rec, prec, f1 = model.k_fold_validation(5,True,True,False)
    print ("Average model accuracy:",acc)
    print ("Average model recall:",rec)
    print ("Average model precision", prec)
    print ("Average F1 score:",f1)


def run_knn_euclidean():
    x_train, y_train = preprocessing(0.01, True) # get data from preprocessing function, will be doing k fold
    model = KNearestNeighbours(33,x_train, y_train)
    # classify data
    acc, rec, prec, f1 = model.k_fold_validation(5, False,True,False)
    print ("Average model accuracy:",acc)
    print ("Average model recall:",rec)
    print ("Average model precision", prec)
    print ("Average F1 score:",f1)

# function that runs experiments with different k values and prints results
def run_k_value_experiments():
    x_train, y_train = preprocessing(0.015, True) # get data from preprocessing function, will be doing k fold
    accuracies = []
    f1_scores = []
    recalls = []
    precisions = []
    for k in range(3,51,2):
        start_time = time.time()

        model = KNearestNeighbours(k,x_train, y_train)
        acc, rec, prec, f1 = model.k_fold_validation(5, True,True,False)
        print ("Accuracy with k value",k,":",acc)
        print ("Average model accuracy:",acc)
        print ("Average model recall:",rec)
        print ("Average model precision", prec)
        print ("Average F1 score:",f1)
        accuracies.append(acc)     
        f1_scores.append(f1)
        recalls.append(rec)
        precisions.append(prec)       
        end_time = time.time()
        print (end_time-start_time)
    plt.plot(range(3,51,2), accuracies,label="accuracy")
    plt.plot(range(3,51,2), recalls,label = "recall")
    plt.plot(range(3,51,2), f1_scores,label = "f1 score")
    plt.plot(range(3,51,2), precisions,label = "precision")
    plt.legend()
    plt.show()
    

# RUN THIS EXPERIMENT TO SEE RESULTS WITH VARYING BALANCING STRATEGIES
def balancing_experiments ():
    x_train, y_train = preprocessing(0.01, True) # get data from preprocessing function, will be doing k fold

    no_balance_model = KNearestNeighbours(33,x_train, y_train)
    acc, rec, prec, f1 = no_balance_model.k_fold_validation(5, True, False, False)
    print ("--------------------------------------------")
    print ("MODEL WITH NO BALANCING")
    print ("Average model accuracy:",acc)
    print ("Average model recall:",rec)
    print ("Average model precision", prec)
    print ("Average F1 score:",f1)

    upsample_model = KNearestNeighbours(33,x_train, y_train)
    acc, rec, prec, f1 = upsample_model.k_fold_validation(5, True, True, True)
    print ("--------------------------------------------")
    print ("MODEL WITH UP SAMPLING")
    print ("Average model accuracy:",acc)
    print ("Average model recall:",rec)
    print ("Average model precision", prec)
    print ("Average F1 score:",f1)

    downsample_model = KNearestNeighbours(33,x_train, y_train)
    acc, rec, prec, f1 = downsample_model.k_fold_validation(5, True, True, False)
    print ("--------------------------------------------")
    print ("MODEL WITH DOWN SAMPLING")
    print ("Average model accuracy:",acc)
    print ("Average model recall:",rec)
    print ("Average model precision", prec)
    print ("Average F1 score:",f1)

def run_ball_tree_experiment():
    x_train, y_train = preprocessing(0.02, True) # get data from preprocessing function, will be doing k fold
    # TIME KNN WITHOUT BALL
    start_time = time.time()
    model_no_ball = KNearestNeighbours(33,x_train, y_train)
    # classify data
    acc, rec, prec, f1 = model_no_ball.k_fold_validation(5, False,True,False)
    print ("Average model accuracy:",acc)
    print ("Average model recall:",rec)
    print ("Average model precision", prec)
    print ("Average F1 score:",f1)
    end_time = time.time()
    print ("TIME FOR KNN EUCLIDEAN:",end_time-start_time)
    # TIME KNN WITHOUT BALL
    start_time = time.time()
    model_ball = KNearestNeighbours(29,x_train, y_train)
    # classify data
    acc, rec, prec, f1 = model_ball.k_fold_validation(33, True,True,False)
    print ("Average model accuracy:",acc)
    print ("Average model recall:",rec)
    print ("Average model precision", prec)
    print ("Average F1 score:",f1)
    end_time = time.time()
    print ("TIME FOR KNN BALL:",end_time-start_time)

#run_ball_tree_experiment()
#balancing_experiments()
#run_k_value_experiments()
run_knn_ball_tree()
