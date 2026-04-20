import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import os
from models_numpy import initialiser_parametres, entrainer, predire, initialiser_mlp, entrainer_mlp, forward_mlp
from utils import taux_erreur

def charger_mnist():
    """Charge et prepare les donnees MNIST"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data_models")
    X_train_path = os.path.join(data_dir, "X_train.npy")
    X_test_path = os.path.join(data_dir, "X_test.npy")
    y_train_path = os.path.join(data_dir, "y_train.npy")
    y_test_path = os.path.join(data_dir, "y_test.npy")
    
    if all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        print("[LOAD] Chargement des donnees MNIST existantes...")
        X_train = np.load(X_train_path)
        X_test = np.load(X_test_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)
    else:
        print("[DOWNLOAD] Telechargement des donnees MNIST...")
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist.data.values, mnist.target.astype(int)
        X = X / 255.0
        
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000].values, y[60000:].values
        
        np.save(X_train_path, X_train)
        np.save(X_test_path, X_test)
        np.save(y_train_path, y_train)
        np.save(y_test_path, y_test)
    
    return X_train, X_test, y_train, y_test


def visualiser_mnist(X_train, y_train):
    """Visualise quelques exemples MNIST"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base_dir, "images", "mnist_exemples.png")
    
    if os.path.exists(img_path):
        print("[OK] mnist_exemples.png existe deja")
        return
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('MNIST Dataset Samples', fontsize=16)
    
    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        ax = axes[i//5, i%5]
        ax.imshow(X_train[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Label: {i}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(img_path, dpi=100)
    plt.show()
    plt.close()
    print("[SAVE] Figure sauvegardee : mnist_exemples.png")


def afficher_distribution(y_train):
    """Affiche la distribution des classes"""
    for digit in range(10):
        count = np.sum(y_train == digit)
        print(f"  Chiffre {digit} : {count} exemples")


def entrainer_modele_lineaire(X_train, y_train, X_test, y_test):
    """Entraine et teste le modele lineaire"""
    print("\n=== MODELE LINEAIRE MNIST ===")
    A, b = initialiser_parametres()
    A, b = entrainer(X_train[:5000], y_train[:5000], A, b, epochs=500)
    
    y_pred_train = predire(X_train, A, b)
    y_pred_test = predire(X_test, A, b)
    
    print("Erreur train :", taux_erreur(y_pred_train, y_train))
    print("Erreur test  :", taux_erreur(y_pred_test, y_test))
    
    return A, b


def entrainer_mlp_mnist(X_train, y_train, X_test, y_test):
    """Entraine et teste le MLP"""
    print("\n=== MLP MNIST ===")
    W1, b1, W2, b2 = initialiser_mlp()
    W1, b1, W2, b2 = entrainer_mlp(
        X_train, y_train,
        W1, b1, W2, b2,
        learning_rate=0.1,
        epochs=50
    )
    
    y_pred_train = np.argmax(forward_mlp(X_train, W1, b1, W2, b2)[0], axis=0)
    y_pred_test = np.argmax(forward_mlp(X_test, W1, b1, W2, b2)[0], axis=0)
    
    print("MLP - Erreur train :", taux_erreur(y_pred_train, y_train))
    print("MLP - Erreur test  :", taux_erreur(y_pred_test, y_test))
    
    return W1, b1, W2, b2


def visualiser_erreurs_mlp(X_test, y_test, W1, b1, W2, b2):
    """Affiche les erreurs du MLP"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base_dir, "images", "erreurs_mlp.png")
    
    if os.path.exists(img_path):
        print("[OK] erreurs_mlp.png existe deja")
        return
    
    y_pred_test = np.argmax(forward_mlp(X_test, W1, b1, W2, b2)[0], axis=0)
    erreurs = np.where(y_pred_test != y_test)[0]
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Exemples mal classes par le MLP", fontsize=13)
    
    for i, idx in enumerate(erreurs[:10]):
        ax = axes[i//5][i%5]
        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Vrai:{y_test[idx]} Predit:{y_pred_test[idx]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(img_path, dpi=100)
    plt.show()
    plt.close()
    print("[SAVE] Figure sauvegardee : erreurs_mlp.png")


def visualiser_tsne(X_test, y_test):
    """Visualise les donnees avec t-SNE"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base_dir, "images", "tsne_mnist.png")
    
    if os.path.exists(img_path):
        print("[OK] tsne_mnist.png existe deja")
        return
    
    X_sample = X_test[:2000]
    y_sample = y_test[:2000]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_sample)
    
    plt.figure(figsize=(10, 8))
    for digit in range(10):
        idx = y_sample == digit
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=str(digit), alpha=0.6, s=10)
    
    plt.legend()
    plt.title("Visualisation t-SNE des donnees MNIST")
    plt.savefig(img_path, dpi=100)
    plt.show()
    plt.close()
    print("[SAVE] Figure sauvegardee : tsne_mnist.png")


def run_mnist():
    """Lance tout le pipeline MNIST"""
    print("\n" + "="*50)
    print("MNIST DATASET")
    print("="*50)
    
    X_train, X_test, y_train, y_test = charger_mnist()
    visualiser_mnist(X_train, y_train)
    afficher_distribution(y_train)
    
    entrainer_modele_lineaire(X_train, y_train, X_test, y_test)
    W1, b1, W2, b2 = entrainer_mlp_mnist(X_train, y_train, X_test, y_test)
    visualiser_erreurs_mlp(X_test, y_test, W1, b1, W2, b2)
    visualiser_tsne(X_test, y_test)
