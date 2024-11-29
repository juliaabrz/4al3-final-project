# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

file_path = 'data/diabetes_012_health_indicators_BRFSS2015.csv'
def preprocessing(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Separate target and features
    target = 'Diabetes_012'
    X = data.drop(columns=[target])
    y = data[target]

    # Normalize numerical features using Min-Max Scaling
    numerical_columns = ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Income' , 'Education', 'GenHealth' ] 
    scaler = MinMaxScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
