import pandas as pd

def load_data(file_path):
    """
    Load the Iris dataset from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file containing the Iris dataset.
    
    Returns:
    DataFrame: A pandas DataFrame containing the loaded dataset.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the Iris dataset.
    
    Parameters:
    data (DataFrame): The raw Iris dataset.
    
    Returns:
    DataFrame: A processed DataFrame ready for analysis.
    """
    # Example preprocessing steps
    # Convert species to categorical
    data['species'] = data['species'].astype('category')
    
    # Handle missing values if any
    data = data.dropna()
    
    return data