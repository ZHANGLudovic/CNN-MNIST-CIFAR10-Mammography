import pandas as pd
import os
import sys


def charger_mass_dataset():
    """Charge le dataset mass case"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "dataset", "mass_case_description_train_set.csv")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} introuvable")
        return None
    
    df = pd.read_csv(csv_path)
    return df


def afficher_info_dataset(df):
    """Affiche les informations du dataset"""
    print("\n=== MASS CASE DATASET ===")
    print("Colonnes :", df.columns.tolist())
    print("\nDistribution pathology :")
    print(df['pathology'].value_counts())
    print("\nNombre total :", len(df))
    print("\nPremières lignes :")
    print(df.head(3))


def explorer_images(df):
    """Explore les chemins des images"""
    print("\nChemin image file path :")
    print(df['image file path'].iloc[0])
    print("\nChemin cropped image file path :")
    print(df['cropped image file path'].iloc[0])


def run_mass_dataset():
    """Lance l'exploration du dataset mass case"""
    df = charger_mass_dataset()
    afficher_info_dataset(df)
    explorer_images(df)
