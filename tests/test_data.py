import unittest
from src.iris_forecasting.data_loader import load_data

class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        data = load_data('data/raw/iris.csv')
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)
        self.assertIn('sepal_length', data.columns)
        self.assertIn('species', data.columns)

if __name__ == '__main__':
    unittest.main()