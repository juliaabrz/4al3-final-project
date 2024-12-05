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

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        y_ = np.where(y <= 0, -1, 1)

        for iteration in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
            
            # testing weights and iterations after every 100 iterations
            # if iteration % 100 == 0:
            #     print(f"Iteration {iteration}: Weight Norm = {np.linalg.norm(self.w)}")

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)



def main():
    X_train, X_test, y_train, y_test = pp.preprocessing()

    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)

    y_pred = svm.predict(X_test)
    accuracy = np.mean(y_pred == np.where(y_test <= 0, -1, 1))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()
