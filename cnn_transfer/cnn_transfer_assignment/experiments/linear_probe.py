"""
Scenario 4.1 – Linear Probe Transfer

"""

import os, random
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ──80/20 split ────────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(root="data/train_data", transform=None)
class_names  = full_dataset.classes

class_to_indices = {}
for idx, (_, label) in enumerate(full_dataset.samples):
    class_to_indices.setdefault(label, []).append(idx)

rng = random.Random(SEED)
train_idx, val_idx = [], []
for label in sorted(class_to_indices):
    idxs = class_to_indices[label][:]
    rng.shuffle(idxs)
    split = int(len(idxs) * 0.8)
    train_idx.extend(idxs[:split])
    val_idx.extend(idxs[split:])

class _SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset; self.indices = indices; self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        return self.transform(img), label

train_dataset = _SubsetWithTransform(full_dataset, train_idx, train_transform)
val_dataset   = _SubsetWithTransform(full_dataset, val_idx,   val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Classes: {len(class_names)}")

# ── Models to run ─────────────────────────────────────────────────────────────
MODELS = {
    "resnet50":       "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny":  "ConvNeXt-Tiny",
}

EPOCHS = 10
all_results = {}

for model_name, model_label in MODELS.items():

    ckpt_path = f"models/{model_name}_linear_probe.pth"

    # Skip if already trained
    if os.path.exists(ckpt_path):
        print(f"\n{'='*60}")
        print(f"Skipping {model_label} — checkpoint already exists at {ckpt_path}")
        print(f"{'='*60}")
        continue

    print(f"\n{'='*60}")
    print(f"Training: {model_label}")
    print(f"{'='*60}")

  
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

   
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier head
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True

    model = model.to(device)

    # Print efficiency metrics
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params   : {total_params - trainable_params:,}")
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, 224, 224).to(device)
        flops = FlopCountAnalysis(model, dummy)
        print(f"MACs (G)        : {flops.total()/1e9:.3f}")
        print(f"FLOPs (G)       : {flops.total()*2/1e9:.3f}\n")
    except:
        print("(fvcore skipped – install for MACs/FLOPs)\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)

    # Training loop
    train_losses, train_accs, val_accs = [], [], []
    all_preds, all_labels_list = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        avg_loss  = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(avg_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        ep_preds, ep_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                ep_preds.extend(preds.cpu().tolist())
                ep_labels.extend(labels.cpu().tolist())

        val_acc = 100 * val_correct / val_total
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        all_preds, all_labels_list = ep_preds, ep_labels

    # Save checkpoint so robustness.py can reuse it
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")

    # Store results
    all_results[model_name] = {
        "train_accs":  train_accs,
        "val_accs":    val_accs,
        "train_losses": train_losses,
        "final_train":  train_accs[-1],
        "final_val":    val_accs[-1],
        "preds":        all_preds,
        "labels":       all_labels_list,
    }

    # ── Plot 1: Accuracy curve ────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS+1), train_accs, "b-o", label="Train Accuracy", markersize=4)
    plt.plot(range(1, EPOCHS+1), val_accs,   "r-o", label="Val Accuracy",   markersize=4)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title(f"Linear Probe – {model_label} Train & Val Accuracy")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_accuracy_curve.png", dpi=150)
    plt.close()

    # ── Plot 2: Loss curve ────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, "b-o", markersize=4)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Linear Probe – {model_label} Training Loss")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_loss_curve.png", dpi=150)
    plt.close()

    # ── Plot 3: Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(all_labels_list, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion Matrix – {model_label} Linear Probe")
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_confusion_matrix.png", dpi=150)
    plt.close()

    # ── Plot 4: PCA + t-SNE embeddings ───────────────────────────────────────
    print(f"\nExtracting features for embedding visualisation …")
    model.eval()
    all_feats, emb_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            feats  = model.forward_features(images)
            if feats.dim() == 4:
                feats = feats.mean(dim=[2, 3])
            elif feats.dim() == 3:
                feats = feats.mean(dim=1)
            all_feats.append(feats.cpu().numpy())
            emb_labels.extend(labels.tolist())

    feats_arr  = np.concatenate(all_feats, axis=0)
    labels_arr = np.array(emb_labels)

    for method, reducer in [("PCA",  PCA(n_components=2, random_state=SEED)),
                             ("tSNE", TSNE(n_components=2, random_state=SEED, perplexity=30))]:
        reduced = reducer.fit_transform(feats_arr)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                              c=labels_arr, cmap="tab20", s=8, alpha=0.7)
        plt.colorbar(scatter, label="Class index")
        plt.title(f"{model_label} – Feature Embeddings ({method})")
        plt.tight_layout()
        plt.savefig(f"plots/{model_name}_feature_{method.lower()}.png", dpi=150)
        plt.close()

    print(f"\n{model_label} done! Final Val Acc: {val_accs[-1]:.2f}%")


print(f"\n{'='*60}")
print("LINEAR PROBE SUMMARY")
print(f"{'='*60}")
for model_name, model_label in MODELS.items():
    if model_name in all_results:
        r = all_results[model_name]
        print(f"{model_label:<20} | Train: {r['final_train']:.2f}% | Val: {r['final_val']:.2f}%")
    else:
        print(f"{model_label:<20} | Skipped (checkpoint exists)")

print("\nAll plots saved to plots/")
print("All checkpoints saved to models/")