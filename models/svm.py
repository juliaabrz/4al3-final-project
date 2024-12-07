from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import preprocessing as pp

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def initialize_weights(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def compute_loss(self, X, y):
        return np.mean(np.maximum(0, 1 - y * (np.dot(X, self.w) + self.b)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)
        y_ = np.where(y <= 0, -1, 1)

        for iteration in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

            if iteration % 100 == 0:
                loss = self.compute_loss(X, y_)
                print(f"Iteration {iteration}: Loss = {loss:.4f}")

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

def evaluate_model(y_true, y_pred):
    y_true_binary = np.where(y_true == -1, 0, 1)
    y_pred_binary = np.where(y_pred == -1, 0, 1)
    acc = accuracy_score(y_true_binary, y_pred_binary)
    return acc

def main():
    X_train, X_test, y_train, y_test = pp.preprocessing()
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_test_ = np.where(y_test <= 0, -1, 1)
    acc = evaluate_model(y_test_, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
