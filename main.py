"""
Main script - Orchestre tout le projet machine learning
"""

import sys
import os

# Ajouter le dossier src au chemin Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Creer les dossiers necessaires s'ils n'existent pas
base_dir = os.path.dirname(__file__)
required_dirs = ['images', 'data_models', 'data', 'dataset']
for dir_name in required_dirs:
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)

from mnist import run_mnist
from convolution import visualiser_filtres_convolution
from cifar10 import run_cifar10
from cnn_torch import run_cnn
from mass_dataset import run_mass_dataset
import numpy as np


def main():
    """Lance le pipeline complet"""
    
    # MNIST 
    run_mnist()
    
    # CONVOLUTIONS 
    print("\n" + "="*50)
    print("CONVOLUTION 2D")
    print("="*50)
    #Charger d'abord les données CIFAR pour avoir des images couleur
    import torchvision
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    X_train = np.array(train_set.data).reshape(-1, 32*32*3) / 255.0
    y_train = np.array(train_set.targets)
    visualiser_filtres_convolution(X_train, y_train)
    
    # CIFAR-10 
    run_cifar10()
    
    # CNN 
    run_cnn()
    
    # MASS CASE DATASET 
    csv_train = os.path.join(base_dir, "dataset", "csv", "mass_case_description_train_set.csv")
    csv_test  = os.path.join(base_dir, "dataset", "csv", "mass_case_description_test_set.csv")
    jpeg_root = os.path.join(base_dir, "dataset", "jpeg")

    run_mass_dataset(csv_train, csv_test, jpeg_root)
    
    print("\n" + "="*50)
    print("Pipeline complet terminé!")
    print("="*50)


if __name__ == "__main__":
    main()
