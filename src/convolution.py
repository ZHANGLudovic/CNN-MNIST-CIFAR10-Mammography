import numpy as np
import matplotlib.pyplot as plt
import os


def convolution2D(image, K, l=0):
    """Applique une convolution 2D sur une image avec un kernel K"""
    H, W = image.shape
    M_pad = np.pad(image, 1)
    M_out = np.zeros((H, W))
    for u in range(H):
        for v in range(W):
            M_out[u, v] = np.sum(K * M_pad[u:u+3, v:v+3]) + l
    return M_out


def visualiser_filtres_convolution(X_train, y_train):
    """Visualise différents filtres de convolution"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base_dir, "images", "filtres_convolution.png")
    
    
    K1 = np.ones((3, 3)) / 9
    K2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    K3 = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
    K4 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    K5 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    K6 = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    
    # Utilise une image chat du dataset CIFAR-10
    idx_chat = np.where(y_train == 3)[0][0]
    img_couleur = X_train[idx_chat].reshape(32, 32, 3)
    img = 0.299*img_couleur[:, :, 0] + 0.587*img_couleur[:, :, 1] + 0.114*img_couleur[:, :, 2]
    
    filtres = [K1, K2, K3, K4, K5, K6]
    noms = ["K1 lissage", "K2 netteté", "K3 bords horiz",
            "K4 bords vert", "K5 Sobel", "K6 diagonal"]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0][0].imshow(img, cmap='gray')
    axes[0][0].set_title("Original")
    axes[0][0].axis('off')
    
    for i, (K, nom) in enumerate(zip(filtres, noms)):
        row, col = (i+1)//4, (i+1)%4
        result = convolution2D(img, K)
        result = np.clip(result, 0, 1)
        axes[row][col].imshow(result, cmap='gray')
        axes[row][col].set_title(nom)
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(img_path, dpi=100)
    plt.show()
    plt.close()
    print("[SAVE] Figure sauvegardee : filtres_convolution.png")
