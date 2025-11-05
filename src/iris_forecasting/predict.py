def load_model(model_path):
    import joblib
    return joblib.load(model_path)

def predict_species(model, features):
    return model.predict(features)

def main(model_path, input_features):
    model = load_model(model_path)
    predictions = predict_species(model, input_features)
    return predictions

if __name__ == "__main__":
    import sys
    import numpy as np

    model_path = sys.argv[1]
    input_features = np.array(sys.argv[2:], dtype=float).reshape(1, -1)
    predictions = main(model_path, input_features)
    print("Predicted species:", predictions)