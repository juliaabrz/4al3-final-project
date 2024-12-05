import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import preprocessing as pp
# import nn
#import knn
#import svm

def main() :
  # here call of the models from the models dir so that we dont have to run a bunch of scripts, this is the main script. but all the work and training and preprocessing is done in each models file
  X_train, X_test, y_train, y_test = pp.preprocessing()

  # call the models
  # nn.neural_network(X_train, y_train, X_test, y_test)
  # knn.knn(X_train, y_train, X_test, y_test)
  # svm.svm(X_train, y_train, X_test, y_test)

main()