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
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)

        # Training loop
        for iteration in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
            
            # Debug output for every 100 iterations
            if iteration % 100 == 0:
                loss = np.mean(np.maximum(0, 1 - y_ * (np.dot(X, self.w) + self.b)))
                print(f"Iteration {iteration}: Loss = {loss:.4f}")

    def predict(self, X):
        # Compute decision boundary
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

def main():
    # Preprocess the data
    X_train, X_test, y_train, y_test = pp.preprocessing()

    # Initialize and train the SVM
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)

    # Make predictions
    y_pred = svm.predict(X_test)

    # Evaluate accuracy
    # Convert y_test to match SVM output format (-1, 1)
    y_test_ = np.where(y_test <= 0, -1, 1)
    accuracy = np.mean(y_pred == y_test_)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Using sklearn accuracy_score for additional confirmation
    y_test_binary = np.where(y_test_ == -1, 0, 1)
    y_pred_binary = np.where(y_pred == -1, 0, 1)
    acc = accuracy_score(y_test_binary, y_pred_binary)
    print(f"Accuracy (Sklearn): {acc:.2f}")

if __name__ == "__main__":
    main()
