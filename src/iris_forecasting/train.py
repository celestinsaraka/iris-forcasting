from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
from iris_forecasting.data_loader import load_data
from iris_forecasting.features import create_features

def train_model():
    # Load the dataset
    data = load_data('data/raw/iris.csv')
    
    # Create features and labels
    X, y = create_features(data)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy:.2f}')
    
    # Save the trained model
    joblib.dump(model, 'models/iris_forecasting_model.pkl')

if __name__ == '__main__':
    train_model()