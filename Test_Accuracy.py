import torch
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from Perceptron_Training import MLP

def evaluate():
    # Load test data (replace with actual data loading logic)
    # Fetch dataset
    mushroom = fetch_ucirepo(id=73)

    # Data preprocessing
    x = mushroom.data.features
    y = mushroom.data.targets.replace({'e': 0, 'p': 1})

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False)
    x_encoded = encoder.fit_transform(x)

    # Standardize the features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_encoded)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.1, random_state=42)

    # Evaluate multiple models
    model_paths = ["models/mushroom_classifier.pth"]
    accuracies = []

    for path in model_paths:
        model_path = "models/mushroom_classifier.pth"
        f = open("data/size.txt", "r")
        model = MLP(int(f.read()))  # Ensure input_size matches model requirements
        model.load_state_dict(torch.load(path))
        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(x_test, dtype=torch.float32))
            predictions = (outputs > 0.5).numpy().astype(int)
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)

    return accuracies[0]
