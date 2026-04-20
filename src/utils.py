import numpy as np

def softmax(z):
    """Calcule la softmax d'une matrice z"""
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def cross_entropy(P, y, n):
    """Calcule la loss de cross-entropy"""
    Y = np.eye(10)[:, y]
    return -np.sum(Y * np.log(P + 1e-9)) / n


def taux_erreur(y_true, y_pred):
    """Calcule le taux d'erreur"""
    return np.mean(y_true != y_pred)
