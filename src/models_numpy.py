import numpy as np
from utils import softmax, cross_entropy

# ============ MODÈLE LINÉAIRE ============

def initialiser_parametres():
    """Initialise les paramètres du modèle linéaire"""
    A = np.random.randn(10, 784) * 0.01
    b = np.random.randn(10, 1) * 0.01
    return A, b


def forward(X, A, b):
    """Forward pass linéaire"""
    z = A @ X.T + b
    P = softmax(z)
    return P


def gradients(X, P, Y, n):
    """Calcule les gradients"""
    dZ = P - Y
    dA = (dZ @ X) / n
    db = np.sum(dZ, axis=1, keepdims=True) / n
    return dA, db


def entrainer(X, y, A, b, learning_rate=0.1, epochs=1000):
    """Entraîne le modèle linéaire"""
    n = X.shape[0]
    Y = np.eye(10)[:, y]
    
    for epoch in range(epochs):
        P = forward(X, A, b)
        cost = cross_entropy(P, y, n)
        dA, db = gradients(X, P, Y, n)
        A -= learning_rate * dA
        b -= learning_rate * db
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    return A, b


def predire(X, A, b):
    """Fait les prédictions"""
    P = forward(X, A, b)
    return np.argmax(P, axis=0)


# ============ MODÈLE MLP ============

def initialiser_mlp():
    """Initialise les paramètres du MLP"""
    W1 = np.random.randn(128, 784) * 0.01
    b1 = np.random.randn(128, 1) * 0.01
    W2 = np.random.randn(10, 128) * 0.01
    b2 = np.random.randn(10, 1) * 0.01
    return W1, b1, W2, b2


def relu(z):
    """Activation ReLU"""
    return np.maximum(0, z)


def forward_mlp(X, W1, b1, W2, b2):
    """Forward pass MLP"""
    Z1 = W1 @ X.T + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    P = softmax(Z2)
    return P, A1, Z1


def backprop_mlp(X, y, P, A1, Z1, W2, n):
    """Backpropagation MLP"""
    Y = np.eye(10)[:, y]
    dZ2 = P - Y
    dW2 = (dZ2 @ A1.T) / n
    db2 = np.sum(dZ2, axis=1, keepdims=True) / n
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (dZ1 @ X) / n
    db1 = np.sum(dZ1, axis=1, keepdims=True) / n
    return dW1, db1, dW2, db2


def entrainer_mlp(X, y, W1, b1, W2, b2, learning_rate, epochs, batch_size=128):
    """Entraîne le MLP"""
    n = X.shape[0]
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffle = X[indices]
        y_shuffle = y[indices]
        
        for i in range(0, n, batch_size):
            X_batch = X_shuffle[i:i+batch_size]
            y_batch = y_shuffle[i:i+batch_size]
            n_batch = X_batch.shape[0]
            
            P, A1, Z1 = forward_mlp(X_batch, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backprop_mlp(X_batch, y_batch, P, A1, Z1, W2, n_batch)
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
        
        if epoch % 10 == 0:
            P_full, _, _ = forward_mlp(X, W1, b1, W2, b2)
            cost = cross_entropy(P_full, y, n)
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    return W1, b1, W2, b2
