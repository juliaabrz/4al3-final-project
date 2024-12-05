import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocessing as pp

# define nn for part 1 of the assignment
class diabetes_neural_network(nn.Module) :
  def __init__(self, features) :
    super(diabetes_neural_network, self).__init__()
    # define the fully connected layers
    # play around with these
    self.fc1 = nn.Linear(features, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 64)  
    self.fc4 = nn.Linear(64, 32)
    self.fc5 = nn.Linear(32, 1)# 0 = no diabetes 1 = prediabetes or diabetes

    # define the pooling layer
    self.pool = nn.MaxPool2d(kernel_size=2)
    
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
    x = F.relu(x)
    # fc
    x = self.fc4(x)
    # relu
    x = F.relu(x)
    # fc
    x = self.fc5(x)
    # outputting as a probability
    x = torch.sigmoid(x)  
    return x
  
def main() :
    # load the dataset
    X_train, X_test, y_train, y_test = pp.preprocessing()

    # get number of features
    features = X_train.shape[1]

    # initialize model
    model = diabetes_neural_network(features)

    # define loss function
    loss_func = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # train
    errors = []
    # Run the algorithm for this many iterations
    EPOCH = 1000
    for _ in range(EPOCH):
        # Apply the model to the test data.
        # Note that this example does not use a holdout set.
        model_outputs = model(X_train)
        # Compute the accumulated loss.
        loss = loss_func(model_outputs, y_train)
        # Record the loss for plotting. This step can be omitted.
        errors.append(loss.item())
        # "Forget" the old gradient from the last iteration.
        optimizer.zero_grad()
        # Back-propagation
        loss.backward()
        # Perform gradient descent
        optimizer.step()

    # plot the losses over epochs
    plt.title("Cross Entropy Loss Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.plot(range(len(errors)), errors)
    plt.show()

    # evaluate
    