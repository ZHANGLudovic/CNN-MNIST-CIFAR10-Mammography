import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
from models_numpy import initialiser_parametres, entrainer, predire, initialiser_mlp, entrainer_mlp, forward_mlp
from utils import taux_erreur


CLASSES = ['avion', 'auto', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion']


def charger_cifar10():
    """Charge et prépare les données CIFAR-10"""
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    
    X_train = np.array(train_set.data).reshape(-1, 32*32*3) / 255.0
    y_train = np.array(train_set.targets)
    X_test = np.array(test_set.data).reshape(-1, 32*32*3) / 255.0
    y_test = np.array(test_set.targets)
    
    print("X_train :", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train :", y_train.shape)
    
    return X_train, X_test, y_train, y_test, train_set, test_set


def visualiser_cifar10(X_train, y_train):
    """Visualise quelques exemples CIFAR-10"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base_dir, "images", "cifar10_samples.png")
    
    if os.path.exists(img_path):
        print("[OK] cifar10_samples.png existe deja")
        return
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        ax = axes[i//5][i%5]
        ax.imshow(X_train[idx].reshape(32, 32, 3))
        ax.set_title(CLASSES[i])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(img_path, dpi=100)
    plt.show()
    plt.close()
    print("[SAVE] Figure sauvegardee : cifar10_samples.png")


def initialiser_parametres_cifar():
    """Initialise les paramètres du modèle linéaire pour CIFAR-10"""
    A = np.random.randn(10, 32*32*3) * 0.01
    b = np.random.randn(10, 1) * 0.01
    return A, b


def entrainer_modele_lineaire_cifar(X_train, y_train, X_test, y_test):
    """Entraîne et teste le modèle linéaire sur CIFAR-10"""
    print("\n=== MODÈLE LINÉAIRE CIFAR-10 ===")
    A_c, b_c = initialiser_parametres_cifar()
    A_c, b_c = entrainer(X_train[:5000], y_train[:5000], A_c, b_c, epochs=500)
    
    y_pred_train_c = predire(X_train, A_c, b_c)
    y_pred_test_c = predire(X_test, A_c, b_c)
    
    print("CIFAR Linéaire — Erreur train :", taux_erreur(y_pred_train_c, y_train))
    print("CIFAR Linéaire — Erreur test  :", taux_erreur(y_pred_test_c, y_test))


def initialiser_mlp_cifar():
    """Initialise les paramètres du MLP pour CIFAR-10"""
    W1 = np.random.randn(128, 32*32*3) * 0.01
    b1 = np.random.randn(128, 1) * 0.01
    W2 = np.random.randn(10, 128) * 0.01
    b2 = np.random.randn(10, 1) * 0.01
    return W1, b1, W2, b2


def entrainer_mlp_cifar(X_train, y_train, X_test, y_test):
    """Entraîne et teste le MLP sur CIFAR-10"""
    print("\n=== MLP CIFAR-10 ===")
    W1_c, b1_c, W2_c, b2_c = initialiser_mlp_cifar()
    W1_c, b1_c, W2_c, b2_c = entrainer_mlp(
        X_train, y_train,
        W1_c, b1_c, W2_c, b2_c,
        learning_rate=0.1,
        epochs=50
    )
    
    y_pred_train_mlp_c = np.argmax(forward_mlp(X_train, W1_c, b1_c, W2_c, b2_c)[0], axis=0)
    y_pred_test_mlp_c = np.argmax(forward_mlp(X_test, W1_c, b1_c, W2_c, b2_c)[0], axis=0)
    
    print("CIFAR MLP — Erreur train :", taux_erreur(y_pred_train_mlp_c, y_train))
    print("CIFAR MLP — Erreur test :", taux_erreur(y_pred_test_mlp_c, y_test))


def run_cifar10():
    """Lance tout le pipeline CIFAR-10"""
    print("\n" + "="*50)
    print("CIFAR-10 DATASET")
    print("="*50)
    
    X_train, X_test, y_train, y_test, train_set, test_set = charger_cifar10()
    visualiser_cifar10(X_train, y_train)
    entrainer_modele_lineaire_cifar(X_train, y_train, X_test, y_test)
    entrainer_mlp_cifar(X_train, y_train, X_test, y_test)
