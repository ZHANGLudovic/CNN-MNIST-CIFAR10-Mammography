import copy
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, classification_report
)
import matplotlib.pyplot as plt

# ============ CONFIGURATION ============
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 25
LR         = 2e-4
PATIENCE   = 6
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)


# ============ CHARGEMENT CSV ============
def pathology_to_label(x):
    x = str(x).strip()
    if x in ["BENIGN", "BENIGN_WITHOUT_CALLBACK"]:
        return 0
    elif x == "MALIGNANT":
        return 1
    return None


def build_image_index(root_dir):
    """Indexe toutes les images jpg du dossier"""
    index_by_filename = {}
    index_by_stem     = {}
    index_by_parent   = {}

    for path in Path(root_dir).rglob("*.jpg"):
        index_by_filename.setdefault(path.name, []).append(str(path))
        index_by_stem.setdefault(path.stem, []).append(str(path))
        index_by_parent.setdefault(path.parent.name, []).append(str(path))

    total = sum(len(v) for v in index_by_filename.values())
    print(f"Total images indexées : {total}")
    return index_by_filename, index_by_stem, index_by_parent


def find_image_path(raw_csv_path, index_by_filename, index_by_stem, index_by_parent):
    """Retrouve le chemin réel d'une image depuis le CSV"""
    if pd.isna(raw_csv_path):
        return None

    parts = str(raw_csv_path).replace("\\", "/").split("/")
    filename = parts[-1] if parts else None
    stem     = Path(filename).stem if filename else None
    parent   = parts[-2] if len(parts) >= 2 else None

    if filename in index_by_filename:
        return index_by_filename[filename][0]
    if stem in index_by_stem:
        return index_by_stem[stem][0]
    if parent in index_by_parent:
        return index_by_parent[parent][0]
    return None


def load_dataframe(csv_path, jpeg_root):
    """Charge le CSV et associe chaque ligne à une image"""
    index_by_filename, index_by_stem, index_by_parent = build_image_index(jpeg_root)

    df = pd.read_csv(csv_path)
    df["label"] = df["pathology"].apply(pathology_to_label)
    df = df[df["label"].notna()].copy()
    
    
    df["img_path"] = df["cropped image file path"].apply(
        lambda x: find_image_path(x, index_by_filename, index_by_stem, index_by_parent)
    )
    df = df[df["img_path"].notna()].reset_index(drop=True)

    print(f"\nCSV : {csv_path}")
    print(f"Nombre d'images trouvées : {len(df)}")
    print("Répartition classes :")
    print(df["label"].value_counts())
    return df


# ============ DATASET ============
def preprocess_mammo(img):
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.convert("RGB")
    return img


train_transform = transforms.Compose([
    transforms.Lambda(preprocess_mammo),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485],
                         std=[0.229, 0.229, 0.229])
])

eval_transform = transforms.Compose([
    transforms.Lambda(preprocess_mammo),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485],
                         std=[0.229, 0.229, 0.229])
])


class MammoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img   = Image.open(self.df.loc[idx, "img_path"])
        label = int(self.df.loc[idx, "label"])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ============ MODÈLE CNN ============
class CNN_Mammographie(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.conv3   = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2   = nn.MaxPool2d(2, 2)
        # 224 -> 112 -> 56 -> 28
        self.fc1     = nn.Linear(64 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2     = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============ ENTRAÎNEMENT ============
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct      += (outputs.argmax(dim=1) == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = correct = total = 0
    all_labels = []
    all_probs  = []
    all_preds  = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            probs   = torch.softmax(outputs, dim=1)[:, 1]
            preds   = outputs.argmax(dim=1)

            running_loss += loss.item() * imgs.size(0)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return (running_loss / total, correct / total, auc,
            np.array(all_labels), np.array(all_probs), np.array(all_preds))


# ============ RUN PRINCIPAL ============
def run_mass_dataset(csv_train, csv_test, jpeg_root):
    print("\n" + "="*50)
    print("MAMMOGRAPHIES — CBIS-DDSM")
    print("="*50)

    # Charger les données d'entraînement
    df_train_full = load_dataframe(csv_train, jpeg_root)
    
    if len(df_train_full) == 0:
        print("❌ Aucune image trouvée pour le train. Vérifiez que les images JPEG sont bien dans :", jpeg_root)
        return
    
    # Charger les données de test si le fichier existe
    import os
    if os.path.exists(csv_test):
        df_test = load_dataframe(csv_test, jpeg_root)
        if len(df_test) == 0:
            print("⚠️  Aucune image trouvée pour le test. Utilisation du set d'entraînement pour évaluation finale.")
            df_test = None
    else:
        print(f"⚠️  Fichier de test non trouvé: {csv_test}")
        df_test = None

    train_df, val_df = train_test_split(
        df_train_full, test_size=0.15,
        stratify=df_train_full["label"], random_state=SEED
    )

    train_dataset = MammoDataset(train_df, transform=train_transform)
    val_dataset   = MammoDataset(val_df,   transform=eval_transform)
    
    if df_test is not None and len(df_test) > 0:
        test_dataset  = MammoDataset(df_test,  transform=eval_transform)
    else:
        test_dataset = None

    # Gestion déséquilibre classes
    train_labels   = train_df["label"].values
    class_counts   = np.bincount(train_labels)
    class_weights  = 1.0 / class_counts
    sample_weights = torch.from_numpy(class_weights[train_labels]).double()

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False) if test_dataset else None

    # Modèle + Loss pondérée
    model   = CNN_Mammographie().to(device)
    n_benin = (train_df["label"] == 0).sum()
    n_malin = (train_df["label"] == 1).sum()
    total   = n_benin + n_malin
    weights = torch.tensor([total/n_benin, total/n_malin], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Boucle entraînement
    best_val_auc = -1.0
    best_state   = None
    counter      = 0
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc, _, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        scheduler.step(val_auc)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = copy.deepcopy(model.state_dict())
            torch.save(best_state, "mammo_checkpoint.pth")
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping déclenché.")
                break

    model.load_state_dict(best_state)

    # Évaluation finale
    if test_loader is not None:
        test_loss, test_acc, test_auc, y_true, y_prob, y_pred = evaluate(
            model, test_loader, criterion
        )

        print("\n===== RÉSULTATS TEST =====")
        print(f"Accuracy : {test_acc:.4f}")
        print(f"AUC      : {test_auc:.4f}")
        print(classification_report(y_true, y_pred,
                                    target_names=["Bénin", "Malin"], digits=4))

        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()
        print(f"Faux négatifs (cancers non détectés) : {FN}")

        disp = ConfusionMatrixDisplay(cm, display_labels=["Bénin", "Malin"])
        disp.plot(cmap="Blues")
        plt.title("Matrice de confusion — Mammographies")
        plt.tight_layout()
        plt.savefig("confusion_matrix_mammo.png", dpi=150)
        plt.show()
    else:
        print("\n===== ENTRAÎNEMENT TERMINÉ (pas de données de test) =====")

    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs_range, history["train_loss"], label="Train loss")
    plt.plot(epochs_range, history["val_loss"],   label="Val loss")
    plt.title("Courbe de loss")
    plt.legend()
    plt.savefig("loss_curve_mammo.png", dpi=150)
    plt.show()

    plt.figure()
    plt.plot(epochs_range, history["train_acc"], label="Train acc")
    plt.plot(epochs_range, history["val_acc"],   label="Val acc")
    plt.title("Courbe d'accuracy")
    plt.legend()
    plt.savefig("accuracy_curve_mammo.png", dpi=150)
    plt.show()

    torch.save(model.state_dict(), "mammo_model_final.pth")
    print("\nModèle sauvegardé : mammo_model_final.pth")