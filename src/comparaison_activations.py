import numpy as np
import matplotlib.pyplot as plt
from models_numpy import initialiser_mlp, entrainer_mlp, forward_mlp
from utils import taux_erreur, cross_entropy, softmax

# ============ FONCTIONS D'ACTIVATION ============

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def heaviside(z):
    return (z >= 0).astype(float)

def relu(z):
    return np.maximum(0, z)

# ============ FORWARD GÉNÉRIQUE ============

def forward_generic(X, W1, b1, W2, b2, activation):
    Z1 = W1 @ X.T + b1
    A1 = activation(Z1)
    Z2 = W2 @ A1 + b2
    P = softmax(Z2)
    return P, A1, Z1

# ============ BACKPROP GÉNÉRIQUE ============

def gradient_activation(activation_name, Z):
    """Calcule le gradient de l'activation"""
    if activation_name == "relu":
        return (Z > 0).astype(float)
    elif activation_name == "sigmoid":
        s = sigmoid(Z)
        return s * (1 - s)  # gradient sigmoid — devient très petit !
    elif activation_name == "heaviside":
        return np.zeros_like(Z)  # gradient nul presque partout !

def backprop_generic(X, y, P, A1, Z1, W2, n, activation_name):
    Y = np.eye(10)[:, y]
    dZ2 = P - Y
    dW2 = (dZ2 @ A1.T) / n
    db2 = np.sum(dZ2, axis=1, keepdims=True) / n
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * gradient_activation(activation_name, Z1)
    dW1 = (dZ1 @ X) / n
    db1 = np.sum(dZ1, axis=1, keepdims=True) / n
    return dW1, db1, dW2, db2

# ============ ENTRAÎNEMENT GÉNÉRIQUE ============

def entrainer_comparaison(X, y, activation_fn, activation_name,
                           epochs=30, batch_size=128, lr=0.1):
    """Entraîne le MLP avec une activation donnée et retourne l'historique des losses"""
    n_total = X.shape[0]
    W1 = np.random.randn(128, 784) * 0.01
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(10, 128) * 0.01
    b2 = np.zeros((10, 1))

    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(n_total)
        X_s, y_s = X[indices], y[indices]

        for i in range(0, n_total, batch_size):
            Xb, yb = X_s[i:i+batch_size], y_s[i:i+batch_size]
            nb = Xb.shape[0]
            P, A1, Z1 = forward_generic(Xb, W1, b1, W2, b2, activation_fn)
            dW1, db1, dW2, db2 = backprop_generic(Xb, yb, P, A1, Z1, W2, nb, activation_name)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        # Loss sur tout le train
        P_full, _, _ = forward_generic(X, W1, b1, W2, b2, activation_fn)
        loss = cross_entropy(P_full, y, n_total)
        losses.append(loss)

        if epoch % 5 == 0:
            print(f"  [{activation_name}] Epoch {epoch} — Loss: {loss:.4f}")

    # Taux d'erreur final
    P_full, _, _ = forward_generic(X, W1, b1, W2, b2, activation_fn)
    y_pred = np.argmax(P_full, axis=0)
    erreur = taux_erreur(y_pred, y)
    print(f"  [{activation_name}] Erreur test : {erreur:.4f}")

    return losses, erreur

# ============ COMPARAISON ============

def run_comparaison_activations(X_train, y_train, X_test, y_test):
    """Compare ReLU, Sigmoid et Heaviside"""
    np.random.seed(42)

    print("\n=== COMPARAISON DES FONCTIONS D'ACTIVATION ===")

    activations = [
        (relu,      "relu",      "ReLU"),
        (sigmoid,   "sigmoid",   "Sigmoid"),
        (heaviside, "heaviside", "Heaviside"),
    ]

    resultats = {}
    couleurs = {"relu": "blue", "sigmoid": "orange", "heaviside": "red"}

    for fn, name, label in activations:
        print(f"\nEntraînement avec {label}...")
        losses, erreur = entrainer_comparaison(
            X_train, y_train, fn, name, epochs=30
        )
        resultats[name] = {"losses": losses, "erreur": erreur, "label": label}

    # ── Courbe de loss ──
    plt.figure(figsize=(10, 5))
    for name, data in resultats.items():
        plt.plot(data["losses"], label=data["label"], color=couleurs[name], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Comparaison des fonctions d'activation — Courbe de loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/comparaison_activations_loss.png", dpi=100)
    plt.show()
    plt.close()

    # ── Tableau des erreurs ──
    print("\n=== RÉSULTATS FINAUX ===")
    print(f"{'Activation':<15} {'Erreur Test':>12}")
    print("-" * 28)
    for name, data in resultats.items():
        print(f"{data['label']:<15} {data['erreur']:>12.4f}")

    # ── Explication visuelle du vanishing gradient ──
    plt.figure(figsize=(10, 4))
    z = np.linspace(-5, 5, 300)
    plt.subplot(1, 2, 1)
    plt.plot(z, relu(z),      label="ReLU",      color="blue",   linewidth=2)
    plt.plot(z, sigmoid(z),   label="Sigmoid",   color="orange", linewidth=2)
    plt.plot(z, heaviside(z), label="Heaviside", color="red",    linewidth=2)
    plt.title("Fonctions d'activation")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    # Gradients
    grad_relu      = (z > 0).astype(float)
    s              = sigmoid(z)
    grad_sigmoid   = s * (1 - s)
    grad_heaviside = np.zeros_like(z)

    plt.plot(z, grad_relu,      label="Gradient ReLU",      color="blue",   linewidth=2)
    plt.plot(z, grad_sigmoid,   label="Gradient Sigmoid",   color="orange", linewidth=2)
    plt.plot(z, grad_heaviside, label="Gradient Heaviside", color="red",    linewidth=2)
    plt.title("Gradients — Vanishing gradient visible")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/comparaison_activations_gradients.png", dpi=100)
    plt.show()
    plt.close()

    print("\n[SAVE] Figures sauvegardées")