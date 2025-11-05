def log_message(message):
    print(f"[LOG] {message}")

def calculate_accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    import joblib
    return joblib.load(filename)