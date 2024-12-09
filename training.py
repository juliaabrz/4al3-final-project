import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score # for evaluating
from torch.utils.data import DataLoader, TensorDataset # for batch training
import numpy as np
from sklearn.model_selection import KFold


def preprocessing(percentage, kfold):
    file_path='diabetes_binary_health_indicators_BRFSS2015.csv'
    # can be downloaded from here: 
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
    correlations = X.corrwith(y)
    selected_features = correlations[correlations.abs() > 0.1].index
    print("Selected features based on correlation:", selected_features.tolist())
    # X = X[selected_features]

    numerical_columns = ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Income' , 'Education', 'GenHlth' ] 
    existing_num_cols = []
    
    for feature in numerical_columns:
        if feature in numerical_columns :
            existing_num_cols.append(feature)
    # Normalize numerical features using Min-Max Scaling
    scaler = MinMaxScaler()
    X[existing_num_cols] = scaler.fit_transform(X[existing_num_cols])

    # X = data.drop(columns=['Income'])

    
    def balance_classes(X, y):
        # count samples
        class_counts = y.value_counts()
        max_count = max(class_counts.values)
        
        # X_balanced and y_balanced store the balanced classes
        X_balanced = []
        y_balanced = []

        for cls in class_counts.keys():
            X_class = X[y == cls]
            y_class = y[y == cls]
            
            if class_counts[cls] < max_count:
                multiplier = max_count // class_counts[cls]
                remainder = max_count % class_counts[cls]
                X_upsampled = pd.concat([X_class] * multiplier + [X_class.sample(remainder, replace=True)])
                y_upsampled = pd.concat([y_class] * multiplier + [y_class.sample(remainder, replace=True)])
            else:
                X_upsampled = X_class
                y_upsampled = y_class
            
            X_balanced.append(X_upsampled)
            y_balanced.append(y_upsampled)
        
        X_balanced = pd.concat(X_balanced)
        y_balanced = pd.concat(y_balanced)
        return X_balanced, y_balanced

    X_balanced, y_balanced = balance_classes(X, y)

    if not kfold:
        # Split the data into training and testing sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    # if for k fold, we want to return all the data
    return X_balanced, y_balanced

# loading training data + pre-processing
X_train, X_test, y_train, y_test = preprocessing(0.1, False)

# training model
##############################
#    Neural Network Model    #
##############################
class diabetes_neural_network(nn.Module) :
  def __init__(self, features) :
    super(diabetes_neural_network, self).__init__()
    # define the fully connected layers
    # play around with these
    self.fc1 = nn.Linear(features, 256)
    self.fc2 = nn.Linear(256, 64)
    self.fc3 = nn.Linear(64, 1)  
    # self.fc4 = nn.Linear(64, 32)
    # self.fc5 = nn.Linear(32, 1)# 0 = no diabetes 1 = prediabetes or diabetes
    self.dropout = nn.Dropout(0.5) # trying regularization
    
  # define the forward pass of the model
  def forward(self, x) :
    # fc
    x = self.fc1(x)
    # relu
    x = F.relu(x)
    # fc
    x = self.fc2(x)
    # relu
    x = F.relu(x)
    # fc
    x = self.fc3(x)
    # relu
    # x = F.relu(x)
    # # fc
    # x = self.fc4(x)
    # # relu
    # x = F.relu(x)
    # # fc
    # x = self.fc5(x)
    # # relu
    # x = F.relu(x)
    # outputting as a probability
    x = self.dropout(x) # regularization
    x = torch.sigmoid(x) # apply sigmoid to get probabilities
    return x
  
# define the training function
def training_nn(model, train_loader, optimizer, loss_func, X_val_tensor, y_val_tensor) :
    print("Training model...")
    training_losses = [] # to store the training losses
    # Run the algorithm for this many iterations
    EPOCH = 200
    for e in range(EPOCH):
        model.train() # sets the model to training mode
        cumulative_loss = 0 # intantiate the cumulative loss

        for x, y in train_loader: # for each batch in the training set
           optimizer.zero_grad() # reset the graident to 0, do this to avoid accumualtion
           outputs = model(x) # predict the probabilities (between 0 and 1)
           loss = loss_func(outputs, y) # calculate the loss
           loss.backward() # backpropagation
           optimizer.step() # perform gradient descent
           cumulative_loss += loss.item() # add the loss to the cumulative loss

        training_losses.append(cumulative_loss / len(train_loader)) # append the average loss for this epoch to the list

        # evaluate on validation set
        model.eval() # sets the model to evaluation mode
        with torch.no_grad(): # no gradient computation during evaluation
            val_outputs = model(X_val_tensor) # predict the probabilities (between 0 and 1)
            val_predictions = (val_outputs >= 0.5).float() # threshold at 0.5 to get binary predictions
            y_val_numpy = y_val_tensor.numpy() # convert tensor to numpy array

            # evaluation metrics
            val_accuracy = accuracy_score(y_val_numpy, val_predictions)
            val_recall = recall_score(y_val_numpy, val_predictions)
            val_f1 = f1_score(y_val_numpy, val_predictions)

        # print the evaluation metrics for every 10th epoch
        if e % 10 == 0 and e != 0:
          print("Current epoch:", e)
          print("\tAvg training loss:", cumulative_loss / len(train_loader))
          print("\tValidation accuracy:", val_accuracy)
          print("\tValidation recall:", val_recall)
          print("\tValidation f1 score:", val_f1)

    # plot the validation losses over epochs
    plt.title("Training Loss Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(EPOCH), training_losses, label="Training Loss")
    plt.legend()
    plt.show()

def neural_network_model(X_train, X_test, y_train, y_test) :
    # split training into train and validation-implent kfold cross validation later
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)#adding a dimension to the tensor to make it compatible with the model
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)#adding a dimension to the tensor to make it compatible with the model
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)#adding a dimension to the tensor to make it compatible with the model

    # create dataloaders for batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # get number of features
    num_features = X_train_tensor.shape[1]

    # initialize model with the number of features
    model = diabetes_neural_network(num_features)

    # define loss function
    loss_func = nn.BCELoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # train the model
    training_nn(model, train_loader, optimizer, loss_func, X_val_tensor, y_val_tensor)

    model_path = "nn.pkl"
    torch.save(model, model_path)

# neural_network_model(X_train, X_test, y_train, y_test)
# svm
##############################
#   Support Vector Machine    #
##############################
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def initialize_weights(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def compute_loss(self, X, y):
        margins = 1 - y * (X.dot(self.w) + self.b)
        return np.mean(np.maximum(0, margins))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        y_ = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        self.initialize_weights(n_features)

        # optimization using stochastic gradient descent
        for iteration in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (x_i.dot(self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - x_i * y_[idx])
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(X.dot(self.w) + self.b)

def evaluate_model(y_true, y_pred):
    y_true_binary = np.where(y_true == -1, 0, 1)
    y_pred_binary = np.where(y_pred == -1, 0, 1)
    acc = accuracy_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    return acc, recall, precision, f1

def svm_model():

    X_train = X_train.astype(float).values
    y_train = y_train.astype(int).values
    X_test = X_test.astype(float).values
    y_test = y_test.astype(int).values

    # k fold cross validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
        svm.fit(X_tr, y_tr)
        y_val_converted = np.where(y_val <= 0, -1, 1)
        y_val_pred = svm.predict(X_val)
        acc = evaluate_model(y_val_converted, y_val_pred)
        fold_accuracies.append(acc)

    avg_acc = np.mean(fold_accuracies)
    print(f"Average validation accuracy across {k} folds: {avg_acc*100:.2f}%")

    final_svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    final_svm.fit(X_train, y_train)