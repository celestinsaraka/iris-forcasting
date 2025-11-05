import pytest
from src.iris_forecasting.predict import make_prediction
from src.iris_forecasting.model import load_model

def test_make_prediction():
    model = load_model('models/trained_model.pkl')
    sample_input = [[5.1, 3.5, 1.4, 0.2]]  # Example input for Iris-setosa
    prediction = make_prediction(model, sample_input)
    assert prediction in ['setosa', 'versicolor', 'virginica']  # Expected classes

def test_invalid_input():
    model = load_model('models/trained_model.pkl')
    invalid_input = [[5.1, 3.5, 'invalid', 0.2]]  # Invalid input
    with pytest.raises(ValueError):
        make_prediction(model, invalid_input)