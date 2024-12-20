import numpy as np
import pickle
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix # for evaluating
import torch
from training import diabetes_neural_network, SVM
import pandas as pd

def evaluate_model(y_true, y_pred):
    y_true_binary = np.where(y_true == -1, 0, 1)
    y_pred_binary = np.where(y_pred == -1, 0, 1)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    print("Final results SVM:\n")
    print(f"Accuracy: {accuracy*100:.1f}%")
    print(f"Recall: {recall*100:.1f}%")
    print(f"Precision: {precision*100:.1f}%")
    print(f"F1 score: {f1*100:.1f}%")

# function to compute male and female bias
def compute_bias(model, X_test_tensor, y_test_tensor, X_test, feature_columns):
    X_test_df = pd.DataFrame(X_test, columns=feature_columns)
    sex_data_tensor = torch.tensor(X_test_df['Sex'].values, dtype=torch.float32)

    # getting predictions from the model
    model.eval()  # setting model to evaluation mode
    with torch.no_grad():
        test_predictions = model(X_test_tensor).squeeze()
        test_predictions = (test_predictions >= 0.5).float()  # threshold at 0.5
    
    male_mask = (sex_data_tensor == 1)  # mask for male
    female_mask = (sex_data_tensor == 0)  # mask for female
    
    # computing tpr and fpr for males
    male_true = y_test_tensor[male_mask]  # true labels for males
    male_pred = test_predictions[male_mask]  # predictions for males
    tn, fp, fn, tp = confusion_matrix(male_true, male_pred).ravel()
    male_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0 #comput tpr
    male_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 #compute fpr

    # computing tpr and fpr for females
    female_true = y_test_tensor[female_mask]  # true labels for females
    female_pred = test_predictions[female_mask]  # predictions for females
    tn, fp, fn, tp = confusion_matrix(female_true, female_pred).ravel()
    female_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0 #comput tpr
    female_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # comput fpr

    # display output
    print(f"Male TPR: {male_tpr:.4f}, Male FPR: {male_fpr:.4f}")
    print(f"Female TPR: {female_tpr:.4f}, Female FPR: {female_fpr:.4f}")

 
# define the evaluation function
def evaluating_nn(model, X_test_tensor, y_test_tensor, X_test, feature_columns) :
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
    print("\nEvaluation Metrics On Held Out Test Set Neural Network")
    print("\tAccuracy:", test_accuracy*100)
    print("\tRecall:", test_recall*100)
    print("\tF1 score:", test_f1*100)

    # call the function that computes the bias
    compute_bias(model, X_test_tensor, y_test_tensor, X_test, feature_columns)

# Helper function for correlation-based feature selection
def apply_correlation_filter(X, y, corr_threshold=0.1):
    correlations = X.corrwith(y)
    selected_features = correlations[correlations.abs() >= corr_threshold].index.tolist()
    selected_features.append('Sex')
    return selected_features
 
def main():
    # GETTING THE FEATUREA-which will later be used for the bias computation
    file_path='diabetes_binary_health_indicators_BRFSS2015.csv'
    # can be downloaded from here: 
    # Load the dataset
    data = pd.read_csv(file_path)
    # drop features that have nan
    data = data.dropna()
    # Separate target and features
    target = 'Diabetes_binary'
    X = data.drop(columns=[target])
    feature_columns = X.columns.tolist()

    # Load model
    with open("svm.pkl", "rb") as f:
        svm_model = pickle.load(f)
 
    # Load test data
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
 
    y_test_converted = np.where(y_test <= 0, -1, 1)
    y_pred = svm_model.predict(X_test)
    # evaluate_model(y_test_converted, y_pred)
 
    # testing neural network
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)#adding a dimension to the tensor to make it compatible with the model
 
    model_path = "nn.pkl"
    loaded_model = torch.load(model_path)
 
    evaluating_nn(loaded_model, X_test_tensor, y_test_tensor, X_test, feature_columns)
 
if __name__ == "__main__":
    main()