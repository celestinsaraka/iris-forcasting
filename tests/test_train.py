import unittest
from src.iris_forecasting.train import train_model
from src.iris_forecasting.data_loader import load_data

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        self.data = load_data('data/raw/iris.csv')
        self.model = train_model(self.data)

    def test_model_training(self):
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'predict'))

    def test_model_accuracy(self):
        accuracy = self.model.evaluate(self.data)
        self.assertGreaterEqual(accuracy, 0.7)  # Assuming we expect at least 70% accuracy

if __name__ == '__main__':
    unittest.main()