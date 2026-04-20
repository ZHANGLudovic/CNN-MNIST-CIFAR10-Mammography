import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from utils import taux_erreur


class CNN(nn.Module):
    """Modèle CNN pour CIFAR-10"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def charger_cifar10_torch():
    """Charge CIFAR-10 et prépare les tenseurs PyTorch"""
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    
    X_tr = torch.tensor(np.array(train_set.data).transpose(0, 3, 1, 2) / 255.0, dtype=torch.float32)
    y_tr = torch.tensor(np.array(train_set.targets), dtype=torch.long)
    X_te = torch.tensor(np.array(test_set.data).transpose(0, 3, 1, 2) / 255.0, dtype=torch.float32)
    y_te = torch.tensor(np.array(test_set.targets), dtype=torch.long)
    
    return X_tr, y_tr, X_te, y_te


def entrainer_cnn(X_tr, y_tr, X_te, y_te, epochs=20, batch_size=128):
    """Entraîne le modèle CNN"""
    print("\n=== CNN CIFAR-10 (PyTorch) ===")
    
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    with torch.no_grad():
        y_pred_cnn = model(X_te).argmax(dim=1).numpy()
    
    print("CNN — Erreur test :", taux_erreur(y_te.numpy(), y_pred_cnn))


def run_cnn():
    """Lance l'entraînement CNN"""
    X_tr, y_tr, X_te, y_te = charger_cifar10_torch()
    entrainer_cnn(X_tr, y_tr, X_te, y_te)
