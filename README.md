# Iris Forecasting Project

This project aims to develop a forecasting model using the Iris dataset. The model will leverage artificial intelligence techniques to analyze and predict outcomes based on the features of the Iris flowers.

## Project Structure

```
iris-forecasting
├── data
│   ├── raw
│   │   └── iris.csv
│   └── processed
├── notebooks
│   └── 01-exploration.ipynb
├── src
│   ├── iris_forecasting
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── features.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── utils.py
│   └── main.py
├── models
├── tests
│   ├── test_data.py
│   ├── test_train.py
│   └── test_predict.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

Make sure you have Python installed on your machine. It is recommended to use a virtual environment to manage dependencies.

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd iris-forecasting
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. To explore the dataset, open the Jupyter notebook located in the `notebooks` directory:
   ```
   jupyter notebook notebooks/01-exploration.ipynb
   ```

2. To train the forecasting model, run the main script:
   ```
   python src/main.py
   ```

3. For predictions, ensure the model is trained and use the prediction functions defined in `src/iris_forecasting/predict.py`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.