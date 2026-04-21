import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)

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

def sensibilite(y_true, y_pred):
    """Taux de vrais positifs (cancers bien détectés)"""
    VP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return VP / (VP + FN)

def specificite(y_true, y_pred):
    """Taux de vrais négatifs"""
    VN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return VN / (VN + FP)

def cross_entropy_binaire(P, y):
    """Loss binaire pour classification bénin/malin"""
    return -np.mean(y * np.log(P + 1e-9) + (1 - y) * np.log(1 - P + 1e-9))