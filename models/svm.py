### svm.py implemenation of the model ###
## By Aniruddh Arora ##

import numpy as np
import pickle
import preprocessing as pp
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

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
                    acc, recall, precision, f1 = evaluate_model(y_val, y_val_pred)
                    print(f"Epoch: {epoch}")
                    print(f"\tAvg training loss: {avg_loss*100:.1f}%")
                    print(f"\tValidation accuracy: {acc*100:.1f}%")
                    print(f"\tValidation recall: {recall*100:.1f}%")
                    print(f"\tValidation precision: {precision*100:.1f}%")
                    print(f"\tValidation F1 score: {f1*100:.1f}%")

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

def main():
    X, y = pp.preprocessing(percentage=0.03, kfold=True)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    # testing different hyperparameters, the regularization parameter is lambda_param
    hyperparams = [
        {'learning_rate': 0.0001, 'lambda_param': 0.001, 'n_iters': 1000},
        {'learning_rate': 0.0005, 'lambda_param': 0.005, 'n_iters': 1000},
        {'learning_rate': 0.001, 'lambda_param': 0.01, 'n_iters': 1000},
    ]

    best_f1 = 0
    best_params = {}
    best_model = None

    for params in hyperparams:
        svm = SVM(learning_rate=params['learning_rate'], lambda_param=params['lambda_param'], n_iters=params['n_iters'])
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_f1s = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            svm.fit(X_tr, y_tr, X_val, y_val, print_epochs=[10, 90])
            y_val_pred = svm.predict(X_val)
            _, _, _, f1 = evaluate_model(y_val, y_val_pred)
            fold_f1s.append(f1)
            # print(f"Fold {fold} F1 Score: {f1*100:.2f}%\n")
        avg_f1 = np.mean(fold_f1s)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_params = params
            best_model = svm

    print("Best Hyperparameters:")
    print(best_params)
    print(f"Best Average Validation F1 Score: {best_f1*100:.2f}%\n")

    X_train_full, X_test, y_train_full, y_test = pp.preprocessing(percentage=0.03, kfold=False)
    X_train_full = np.asarray(X_train_full, dtype=float)
    y_train_full = np.asarray(y_train_full, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=int)

    # train and evaluate the best model on the full training set
    best_model.fit(X_train_full, y_train_full, print_epochs=[10, 90])
    y_test_pred = best_model.predict(X_test)
    test_acc, test_recall, test_precision, test_f1 = evaluate_model(y_test, y_test_pred)
    print("Final results:")
    print(f"Accuracy: {test_acc*100:.1f}%")
    print(f"Recall: {test_recall*100:.1f}%")
    print(f"Precision: {test_precision*100:.1f}%")
    print(f"F1 score: {test_f1*100:.1f}%")

    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    np.save("test_features.npy", X_test)
    np.save("test_labels.npy", y_test)

if __name__ == "__main__":
    main()

