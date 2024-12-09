import numpy as np
import pickle
from sklearn.metrics import accuracy_score, recall_score, f1_score # for evaluating
import torch
from training import diabetes_neural_network, SVM

def evaluate_model(y_true, y_pred):
    y_true_binary = np.where(y_true == -1, 0, 1)
    y_pred_binary = np.where(y_pred == -1, 0, 1)
    return accuracy_score(y_true_binary, y_pred_binary)

# define the evaluation function
def evaluating_nn(model, X_test_tensor, y_test_tensor) :
    # evaluate on held out test set
    model.eval() # sets the model to evaluation mode
    with torch.no_grad(): # no gradient computation during evaluation
        test_outputs = model(X_test_tensor) # predict the probabilities (between 0 and 1)
        test_predictions = (test_outputs >= 0.5).float() # threshold at 0.5 to get binary predictions
        y_test_numpy = y_test_tensor.numpy() # convert tensor to numpy array

        # evaluation metrics
        test_accuracy = accuracy_score(y_test_numpy, test_predictions)
        test_recall = recall_score(y_test_numpy, test_predictions)
        test_f1 = f1_score(y_test_numpy, test_predictions)

    # print the evaluation metrics
    print("\nEvaluation Metrics On Held Out Test Set")
    print("\tAccuracy:", test_accuracy*100)
    print("\tRecall:", test_recall*100)
    print("\tF1 score:", test_f1*100)

def main():
    # Load model
    with open("svm.pkl", "rb") as f:
        svm_model = pickle.load(f)

    # Load test data
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    y_test_converted = np.where(y_test <= 0, -1, 1)
    y_pred = svm_model.predict(X_test)
    acc = evaluate_model(y_test_converted, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")

    # testing nerual network
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)#adding a dimension to the tensor to make it compatible with the model

    model_path = "nn.pkl"
    loaded_model = torch.load(model_path)

    evaluating_nn(loaded_model, X_test_tensor, y_test_tensor)

if __name__ == "__main__":
    main()
