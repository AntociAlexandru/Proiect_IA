import tkinter as tk
from tkinter import ttk
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from ucimlrepo import fetch_ucirepo
from Perceptron_Training import MLP, train
from Test_Accuracy import evaluate
import numpy as np
import pandas as pd

# Load attributes from JSON file
with open("data/attributes.json", "r") as f:
    attributes = json.load(f)

# Load trained model
model_path = "models/mushroom_classifier.pth"
with open("data/size.txt", "r") as f:
    input_size = int(f.read())

model = MLP(input_size)  # Ensure input_size matches the model's requirements
model.load_state_dict(torch.load(model_path))
model.eval()

# Load preprocessing objects
mushroom = fetch_ucirepo(id=73)
data_features = mushroom.data.features
encoder = OneHotEncoder(sparse_output=False)
scaler = StandardScaler()

# Fit encoder and scaler with the dataset
encoded_features = encoder.fit_transform(data_features)
scaler.fit(encoded_features)


def classify():
    inputs = []
    for key, combo in combos.items():
        value = combo.get()
        inputs.append(value)

    # Convert inputs to DataFrame
    input_df = pd.DataFrame([inputs], columns=attributes.keys())

    # Preprocess the input using pre-fitted encoder and scaler
    encoded_input = encoder.transform(input_df)
    processed_input = scaler.transform(encoded_input)

    # Ensure input tensor shape matches model's requirements
    with torch.no_grad():
        input_tensor = torch.tensor(processed_input, dtype=torch.float32)
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()  # Assuming binary classification with sigmoid activation

    probability_label.config(text=f"Probability: {probability:.2%}")


def train_n_load():
    train()

    # Load trained model
    model_path = "models/mushroom_classifier.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

def get_acc():
    acc = evaluate()
    accuracy_label.config(text=f"Accuracy: {acc:.2%}")

# Create the main application window
root = tk.Tk()
root.title("Mushroom Classifier")
combos = {}

attributes_frame = tk.Frame(root)
attributes_frame.pack(padx=10, pady=10)

for i, (key, values) in enumerate(attributes.items()):
    label = tk.Label(attributes_frame, text=key.replace("_", " ").capitalize() + ":")
    label.grid(row=i, column=0, sticky="w", padx=5, pady=2)

    combo = ttk.Combobox(attributes_frame, values=list(values.values()))
    combo.grid(row=i, column=1, padx=5, pady=2)
    combo.set(list(values.values())[0])
    combos[key] = combo

bottom_frame = tk.Frame(root)
bottom_frame.pack(padx=10, pady=10)

probability_label = tk.Label(bottom_frame, text="Probability: Unknown", font=("Arial", 12, "bold"))
probability_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

accuracy_label = tk.Label(bottom_frame, text="Accuracy: Unknown", font=("Arial", 12, "bold"))
accuracy_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

classify_button = tk.Button(bottom_frame, text="Classify", command=classify)
classify_button.grid(row=2, column=0, padx=10, pady=5)

train_button = tk.Button(bottom_frame, text="Train", command=train_n_load)
train_button.grid(row=2, column=1, padx=10, pady=5)

eval_button = tk.Button(bottom_frame, text="Evaluate", command=get_acc)
eval_button.grid(row=2, column=2, padx=10, pady=5)

exit_button = tk.Button(bottom_frame, text="Exit", command=root.quit)
exit_button.grid(row=2, column=3, padx=10, pady=5)

root.mainloop()
