import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
from models_numpy import initialiser_parametres, entrainer, predire, initialiser_mlp, entrainer_mlp, forward_mlp, initialiser_mlp2, entrainer_mlp2, forward_mlp2
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


def initialiser_mlp2_cifar():
    """Initialise les paramètres du MLP 2 couches pour CIFAR-10 couleur"""
    W1 = np.random.randn(128, 32*32*3) * 0.01
    b1 = np.random.randn(128, 1) * 0.01
    W2 = np.random.randn(64, 128) * 0.01
    b2 = np.random.randn(64, 1) * 0.01
    W3 = np.random.randn(10, 64) * 0.01
    b3 = np.random.randn(10, 1) * 0.01
    return W1, b1, W2, b2, W3, b3


def entrainer_mlp2_cifar(X_train, y_train, X_test, y_test):
    """Entraîne et teste le MLP 2 couches sur CIFAR-10"""
    print("\n=== MLP 2 COUCHES CIFAR-10 ===")
    W1_c, b1_c, W2_c, b2_c, W3_c, b3_c = initialiser_mlp2_cifar()
    W1_c, b1_c, W2_c, b2_c, W3_c, b3_c = entrainer_mlp2(
        X_train, y_train,
        W1_c, b1_c, W2_c, b2_c, W3_c, b3_c,
        learning_rate=0.1,
        epochs=50
    )
    
    y_pred_train_mlp2_c = np.argmax(forward_mlp2(X_train, W1_c, b1_c, W2_c, b2_c, W3_c, b3_c)[0], axis=0)
    y_pred_test_mlp2_c = np.argmax(forward_mlp2(X_test, W1_c, b1_c, W2_c, b2_c, W3_c, b3_c)[0], axis=0)
    
    print("CIFAR MLP 2 couches — Erreur train :", taux_erreur(y_pred_train_mlp2_c, y_train))
    print("CIFAR MLP 2 couches — Erreur test :", taux_erreur(y_pred_test_mlp2_c, y_test))
    
def convertir_grayscale(train_set, test_set):
    """Convertit les images CIFAR-10 en niveaux de gris"""
    X_train_raw = np.array(train_set.data) / 255.0  # (50000, 32, 32, 3)
    X_test_raw  = np.array(test_set.data)  / 255.0  # (10000, 32, 32, 3)

    # Formule standard de conversion
    X_train_gray = (0.299 * X_train_raw[:,:,:,0] +
                    0.587 * X_train_raw[:,:,:,1] +
                    0.114 * X_train_raw[:,:,:,2])

    X_test_gray  = (0.299 * X_test_raw[:,:,:,0] +
                    0.587 * X_test_raw[:,:,:,1] +
                    0.114 * X_test_raw[:,:,:,2])

    # Aplatir en vecteurs 1024
    X_train_gray = X_train_gray.reshape(-1, 1024)
    X_test_gray  = X_test_gray.reshape(-1, 1024)

    print("X_train_gray :", X_train_gray.shape)
    print("X_test_gray  :", X_test_gray.shape)

    return X_train_gray, X_test_gray


def entrainer_modele_lineaire_cifar_gray(X_train_gray, y_train, X_test_gray, y_test):
    """Entraîne le modèle linéaire sur CIFAR-10 en niveaux de gris"""
    print("\n=== MODÈLE LINÉAIRE CIFAR-10 NIVEAUX DE GRIS ===")
    A = np.random.randn(10, 1024) * 0.01
    b = np.random.randn(10, 1) * 0.01
    A, b = entrainer(X_train_gray[:5000], y_train[:5000], A, b, epochs=500)

    y_pred_train = predire(X_train_gray, A, b)
    y_pred_test  = predire(X_test_gray,  A, b)

    print("Linéaire Gris — Erreur train :", taux_erreur(y_pred_train, y_train))
    print("Linéaire Gris — Erreur test  :", taux_erreur(y_pred_test,  y_test))

    return A, b


def entrainer_mlp_cifar_gray(X_train_gray, y_train, X_test_gray, y_test):
    """Entraîne le MLP sur CIFAR-10 en niveaux de gris"""
    print("\n=== MLP CIFAR-10 NIVEAUX DE GRIS ===")
    W1 = np.random.randn(128, 1024) * 0.01
    b1 = np.random.randn(128, 1) * 0.01
    W2 = np.random.randn(10, 128) * 0.01
    b2 = np.random.randn(10, 1) * 0.01

    W1, b1, W2, b2 = entrainer_mlp(
        X_train_gray, y_train,
        W1, b1, W2, b2,
        learning_rate=0.1,
        epochs=50
    )

    y_pred_train = np.argmax(forward_mlp(X_train_gray, W1, b1, W2, b2)[0], axis=0)
    y_pred_test  = np.argmax(forward_mlp(X_test_gray,  W1, b1, W2, b2)[0], axis=0)

    print("MLP Gris — Erreur train :", taux_erreur(y_pred_train, y_train))
    print("MLP Gris — Erreur test  :", taux_erreur(y_pred_test,  y_test))


def initialiser_mlp2_cifar_gray():
    """Initialise les paramètres du MLP 2 couches pour CIFAR-10 gris"""
    W1 = np.random.randn(128, 1024) * 0.01
    b1 = np.random.randn(128, 1) * 0.01
    W2 = np.random.randn(64, 128) * 0.01
    b2 = np.random.randn(64, 1) * 0.01
    W3 = np.random.randn(10, 64) * 0.01
    b3 = np.random.randn(10, 1) * 0.01
    return W1, b1, W2, b2, W3, b3


def entrainer_mlp2_cifar_gray(X_train_gray, y_train, X_test_gray, y_test):
    """Entraîne le MLP 2 couches sur CIFAR-10 en niveaux de gris"""
    print("\n=== MLP 2 COUCHES CIFAR-10 NIVEAUX DE GRIS ===")
    W1, b1, W2, b2, W3, b3 = initialiser_mlp2_cifar_gray()
    W1, b1, W2, b2, W3, b3 = entrainer_mlp2(
        X_train_gray, y_train,
        W1, b1, W2, b2, W3, b3,
        learning_rate=0.1,
        epochs=50
    )

    y_pred_train = np.argmax(forward_mlp2(X_train_gray, W1, b1, W2, b2, W3, b3)[0], axis=0)
    y_pred_test  = np.argmax(forward_mlp2(X_test_gray, W1, b1, W2, b2, W3, b3)[0], axis=0)

    print("MLP 2 couches Gris — Erreur train :", taux_erreur(y_pred_train, y_train))
    print("MLP 2 couches Gris — Erreur test  :", taux_erreur(y_pred_test,  y_test))


def comparer_cifar(X_train, y_train, X_test, y_test,
                   X_train_gray, X_test_gray):
    """Compare les résultats gris vs couleur"""
    print("\n=== COMPARAISON GRIS VS COULEUR CIFAR-10 ===")

    # Linéaire couleur
    A_c, b_c = initialiser_parametres_cifar()
    A_c, b_c = entrainer(X_train[:5000], y_train[:5000], A_c, b_c, epochs=500)
    err_test_lin_color = taux_erreur(predire(X_test, A_c, b_c), y_test)

    # Linéaire gris
    A_g = np.random.randn(10, 1024) * 0.01
    b_g = np.random.randn(10, 1) * 0.01
    A_g, b_g = entrainer(X_train_gray[:5000], y_train[:5000], A_g, b_g, epochs=500)
    err_test_lin_gray = taux_erreur(predire(X_test_gray, A_g, b_g), y_test)

    # MLP 1 couche couleur
    W1_c, b1_c, W2_c, b2_c = initialiser_mlp_cifar()
    W1_c, b1_c, W2_c, b2_c = entrainer_mlp(X_train, y_train, W1_c, b1_c, W2_c, b2_c,
                                             learning_rate=0.1, epochs=50)
    err_test_mlp_color = taux_erreur(
        np.argmax(forward_mlp(X_test, W1_c, b1_c, W2_c, b2_c)[0], axis=0), y_test)

    # MLP 1 couche gris
    W1_g = np.random.randn(128, 1024) * 0.01
    b1_g = np.random.randn(128, 1) * 0.01
    W2_g = np.random.randn(10, 128) * 0.01
    b2_g = np.random.randn(10, 1) * 0.01
    W1_g, b1_g, W2_g, b2_g = entrainer_mlp(X_train_gray, y_train, W1_g, b1_g, W2_g, b2_g,
                                             learning_rate=0.1, epochs=50)
    err_test_mlp_gray = taux_erreur(
        np.argmax(forward_mlp(X_test_gray, W1_g, b1_g, W2_g, b2_g)[0], axis=0), y_test)

    # MLP 2 couches couleur
    W1_c2, b1_c2, W2_c2, b2_c2, W3_c2, b3_c2 = initialiser_mlp2_cifar()
    W1_c2, b1_c2, W2_c2, b2_c2, W3_c2, b3_c2 = entrainer_mlp2(
        X_train, y_train, W1_c2, b1_c2, W2_c2, b2_c2, W3_c2, b3_c2,
        learning_rate=0.1, epochs=50)
    err_test_mlp2_color = taux_erreur(
        np.argmax(forward_mlp2(X_test, W1_c2, b1_c2, W2_c2, b2_c2, W3_c2, b3_c2)[0], axis=0), y_test)

    # MLP 2 couches gris
    W1_g2, b1_g2, W2_g2, b2_g2, W3_g2, b3_g2 = initialiser_mlp2_cifar_gray()
    W1_g2, b1_g2, W2_g2, b2_g2, W3_g2, b3_g2 = entrainer_mlp2(
        X_train_gray, y_train, W1_g2, b1_g2, W2_g2, b2_g2, W3_g2, b3_g2,
        learning_rate=0.1, epochs=50)
    err_test_mlp2_gray = taux_erreur(
        np.argmax(forward_mlp2(X_test_gray, W1_g2, b1_g2, W2_g2, b2_g2, W3_g2, b3_g2)[0], axis=0), y_test)

    print(f"\n{'Modèle':<25} {'Gris':>10} {'Couleur':>10}")
    print("-" * 45)
    print(f"{'Linéaire':<25} {err_test_lin_gray:>10.4f} {err_test_lin_color:>10.4f}")
    print(f"{'MLP 1 couche':<25} {err_test_mlp_gray:>10.4f} {err_test_mlp_color:>10.4f}")
    print(f"{'MLP 2 couches':<25} {err_test_mlp2_gray:>10.4f} {err_test_mlp2_color:>10.4f}")


def run_cifar10():
    """Lance tout le pipeline CIFAR-10"""
    print("\n" + "="*50)
    print("CIFAR-10 DATASET")
    print("="*50)

    X_train, X_test, y_train, y_test, train_set, test_set = charger_cifar10()
    visualiser_cifar10(X_train, y_train)

    # Version couleur
    entrainer_modele_lineaire_cifar(X_train, y_train, X_test, y_test)
    entrainer_mlp_cifar(X_train, y_train, X_test, y_test)

    # Version niveaux de gris
    X_train_gray, X_test_gray = convertir_grayscale(train_set, test_set)  
    entrainer_modele_lineaire_cifar_gray(X_train_gray, y_train, X_test_gray, y_test)  
    entrainer_mlp_cifar_gray(X_train_gray, y_train, X_test_gray, y_test)  

    # Comparaison
    comparer_cifar(X_train, y_train, X_test, y_test,
                   X_train_gray, X_test_gray)  
