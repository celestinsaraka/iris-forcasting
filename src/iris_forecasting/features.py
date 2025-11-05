def create_features(df):
    # Example feature engineering: creating a new feature based on existing ones
    df['sepal_area'] = df['sepal_length'] * df['sepal_width']
    df['petal_area'] = df['petal_length'] * df['petal_width']
    
    # Selecting relevant features for the forecasting model
    features = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'sepal_area', 'petal_area']]
    
    return features

def encode_target(df):
    # Example encoding of target variable (species)
    df['species'] = df['species'].astype('category').cat.codes
    return df['species']