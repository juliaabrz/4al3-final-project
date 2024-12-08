### training.py Implementation of SVM Model using Hinge loss and SGD ###
## By Aniruddh Arora ##

import numpy as np
import pandas as pd
import pickle
import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

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

    def compute_loss(self, x_i, y_i):
        return max(0, 1 - y_i * (x_i.dot(self.w) + self.b))

    def fit(self, X, y, X_val=None, y_val=None, print_epochs=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        y_ = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        self.initialize_weights(n_features)

        if print_epochs is None:
            print_epochs = []

        for epoch in range(1, self.n_iters + 1):
            epoch_loss = 0
            for idx in range(n_samples):
                x_i = X[idx]
                y_i = y_[idx]
                condition = y_i * (x_i.dot(self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - x_i * y_i)
                    self.b -= self.lr * y_i
                loss = self.compute_loss(x_i, y_i)
                epoch_loss += loss
            avg_loss = epoch_loss / n_samples

            if epoch in print_epochs:
                if X_val is not None and y_val is not None:
                    y_val_pred = self.predict(X_val)
                    acc, recall, f1 = evaluate_model(y_val, y_val_pred)
                    print(f"Epoch: {epoch}")
                    print(f"\tAvg training loss: {avg_loss*100:.1f}%")
                    print(f"\tValidation accuracy: {acc*100:.1f}%")
                    print(f"\tValidation recall: {recall*100:.1f}%")
                    print(f"\tValidation F1 score: {f1*100:.1f}%")

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(X.dot(self.w) + self.b)

def evaluate_model(y_true, y_pred):
    y_true_binary = np.where(y_true == -1, 0, 1)
    y_pred_binary = np.where(y_pred == -1, 0, 1)
    acc = accuracy_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    return acc, recall, f1

def main():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = pp.preprocessing(percentage=0.01, kfold=False)
    X_train = X_train.astype(float).values
    y_train = y_train.astype(int).values
    X_test = X_test.astype(float).values
    y_test = y_test.astype(int).values

    # Split training data into training and validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize and train the SVM model
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_tr, y_tr, X_val, y_val, print_epochs=[10, 90])

    # Final evaluation on the test set
    y_test_pred = svm.predict(X_test)
    test_acc, test_recall, test_f1 = evaluate_model(y_test, y_test_pred)
    print("\nFinal results:")
    print(f"Accuracy: {test_acc*100:.1f}%")
    print(f"Recall: {test_recall*100:.1f}%")
    print(f"F1 score: {test_f1*100:.1f}%")

    # Save the trained model and test data
    with open("model.pkl", "wb") as f:
        pickle.dump(svm, f)
    np.save("test_features.npy", X_test)
    np.save("test_labels.npy", y_test)

if __name__ == "__main__":
    main()
