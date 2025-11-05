import pandas as pd
from iris_forecasting.data_loader import load_data
from iris_forecasting.train import train_model
from iris_forecasting.predict import make_predictions

def main():
    # Load the Iris dataset
    data = load_data('data/raw/iris.csv')
    
    # Train the forecasting model
    model = train_model(data)
    
    # Make predictions
    predictions = make_predictions(model, data)
    
    # Output predictions
    print(predictions)

if __name__ == "__main__":
    main()