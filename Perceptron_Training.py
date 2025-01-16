import torch
import torch.nn as nn
import torch.optim as optim
#import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from EvolutionaryAlgorithm import EvolutionaryAlgorithm, Chromosome, IOptimizationProblem


# Define MLP
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, input_size // 2)
        self.hidden_layer = nn.Linear(input_size // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.sigmoid(self.input_layer(x))
        x = self.sigmoid(self.hidden_layer(x))
        return x

# Define a fitness function for the evolutionary algorithm
class MLPEvolutionProblem(IOptimizationProblem):
    def __init__(self, model, criterion, x_train, y_train):
        self.model = model
        self.criterion = criterion
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train.values, dtype=torch.float32)

    def make_chromosome(self):
        # Create a chromosome representing model parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        return Chromosome(num_params, num_params*[-10000], num_params*[10000])

    def compute_fitness(self, chromosome):
        # Assign the chromosome's genes to model parameters
        with torch.no_grad():
            idx = 0
            for param in self.model.parameters():
                numel = param.numel()
                param.data = torch.tensor(chromosome.genes[idx:idx + numel], dtype=torch.float32).view(param.size())
                idx += numel

        # Compute loss
        outputs = self.model(self.x_train)
        loss = self.criterion(outputs, self.y_train)

        # Fitness is the negative loss (maximize fitness = minimize loss)
        chromosome.fitness = -loss.item()

def train(model_path = "models/mushroom_classifier.pth"):
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

    # Initialize model, loss, and evolutionary algorithm
    input_size = x_train.shape[1]
    with open("data/size.txt", 'w') as file:
        file.write(str(input_size))
    model = MLP(input_size)
    criterion = nn.BCELoss()
    problem = MLPEvolutionProblem(model, criterion, x_train, y_train)

    ea = EvolutionaryAlgorithm()

    # Solve the optimization problem using evolutionary algorithm
    solution = ea.solve(problem, population_size=50, max_generations=100, crossover_rate=0.9, mutation_rate=0.1)

    # Assign the best solution to the model parameters
    with torch.no_grad():
        idx = 0
        for param in model.parameters():
            numel = param.numel()
            param.data = torch.tensor(solution.genes[idx:idx + numel], dtype=torch.float32).view(param.size())
            idx += numel

    torch.save(model.state_dict(), model_path)

train()