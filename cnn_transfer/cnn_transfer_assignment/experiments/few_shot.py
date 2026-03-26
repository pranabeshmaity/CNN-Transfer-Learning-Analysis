

import os, random
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("plots", exist_ok=True)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(root="data/train_data", transform=None)
class_names  = full_dataset.classes

class_to_indices = {}
for idx, (_, label) in enumerate(full_dataset.samples):
    class_to_indices.setdefault(label, []).append(idx)

rng_split = random.Random(SEED)
train_idx, val_idx = [], []
for label in sorted(class_to_indices):
    idxs = class_to_indices[label][:]
    rng_split.shuffle(idxs)
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

full_train_dataset = _SubsetWithTransform(full_dataset, train_idx, train_transform)
val_dataset        = _SubsetWithTransform(full_dataset, val_idx,   val_transform)
val_loader         = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

def create_stratified_subset(dataset, percent, seed=SEED):
    if percent >= 1.0:
        return dataset
    rng = random.Random(seed)
    label_map = {}
    for pos in range(len(dataset)):
        lbl = dataset.dataset.targets[dataset.indices[pos]]
        label_map.setdefault(lbl, []).append(pos)
    selected = []
    for lbl in sorted(label_map):
        pool = label_map[lbl][:]
        rng.shuffle(pool)
        k = max(1, int(len(pool) * percent))
        selected.extend(pool[:k])
    return Subset(dataset, selected)

def evaluate_model(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total

def run_experiment(model_name, data_percent, max_epochs=15):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    subset = create_stratified_subset(full_train_dataset, data_percent)
    loader = DataLoader(subset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    print(f"\n  {int(data_percent*100)}% → {len(subset)} train samples")

    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
    for p in model.parameters(): p.requires_grad = False
    for p in model.get_classifier().parameters(): p.requires_grad = True
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.get_classifier().parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = max_epochs if data_percent < 1.0 else 20
    val_acc_curve, train_acc_curve = [], []

    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            criterion(outputs, labels).backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)

        train_acc = 100 * correct / total
        val_acc   = evaluate_model(model)
        val_acc_curve.append(val_acc)
        train_acc_curve.append(train_acc)
        print(f"  Epoch {epoch+1:02d}/{epochs} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    best_val  = max(val_acc_curve)
    final_gap = train_acc_curve[-1] - val_acc_curve[-1]
    print(f"  Best Val: {best_val:.2f}% | Train-Val Gap: {final_gap:.2f}%")

    return {
        "best_val_acc":    best_val,
        "final_train_acc": train_acc_curve[-1],
        "final_val_acc":   val_acc_curve[-1],
        "train_val_gap":   final_gap,
        "val_curve":       val_acc_curve,
        "train_curve":     train_acc_curve,
    }

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "resnet50":        "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny":   "ConvNeXt-Tiny",
}

data_levels = [1.0, 0.20, 0.05]
all_results = {}

for model_name, model_label in MODELS.items():
    print(f"\n{'#'*60}")
    print(f"MODEL: {model_label}")
    print(f"{'#'*60}")

    results = {}
    for level in data_levels:
        key = f"{int(level*100)}pct"
        results[key] = run_experiment(model_name, level)

    all_results[model_name] = results

    acc100 = results["100pct"]["best_val_acc"]
    acc5   = results["5pct"]["best_val_acc"]
    delta  = (acc100 - acc5) / acc100

    print(f"\nRelative Performance Drop Δ = {delta*100:.2f}%")
    print(f"  Acc@100%: {acc100:.2f}% | Acc@5%: {acc5:.2f}%")
    print(f"Train-Val Gaps:")
    for k, r in results.items():
        print(f"  {k}: {r['train_val_gap']:.2f}%")

    # Plot 1: Data efficiency line plot
    pcts = [5, 20, 100]
    accs = [results["5pct"]["best_val_acc"],
            results["20pct"]["best_val_acc"],
            results["100pct"]["best_val_acc"]]

    plt.figure(figsize=(7, 5))
    plt.plot(pcts, accs, "o-", linewidth=2, markersize=8, color="tab:blue")
    for x, y in zip(pcts, accs):
        plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(5, 6), fontsize=9)
    plt.xlabel("Training Data Used (%)"); plt.ylabel("Best Val Accuracy (%)")
    plt.title(f"{model_label} – Data Efficiency")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_few_shot_results.png", dpi=150)
    plt.close()

    # Plot 2: Val accuracy curves
    plt.figure(figsize=(9, 5))
    for key, col in zip(["5pct","20pct","100pct"], ["tab:red","tab:orange","tab:blue"]):
        curve = results[key]["val_curve"]
        plt.plot(range(1, len(curve)+1), curve, label=key, color=col)
    plt.xlabel("Epoch"); plt.ylabel("Val Accuracy (%)")
    plt.title(f"{model_label} – Few-Shot Val Accuracy Curves")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_few_shot_curves.png", dpi=150)
    plt.close()


print(f"\n{'='*60}")
print("FEW-SHOT FULL SUMMARY")
print(f"{'='*60}")
for model_name, model_label in MODELS.items():
    r = all_results[model_name]
    acc100 = r["100pct"]["best_val_acc"]
    acc20  = r["20pct"]["best_val_acc"]
    acc5   = r["5pct"]["best_val_acc"]
    delta  = (acc100 - acc5) / acc100 * 100
    print(f"\n{model_label}:")
    print(f"  100%: {acc100:.2f}% | 20%: {acc20:.2f}% | 5%: {acc5:.2f}% | Δ={delta:.2f}%")
    print(f"  Gaps → 100%: {r['100pct']['train_val_gap']:.2f}% | "
          f"20%: {r['20pct']['train_val_gap']:.2f}% | "
          f"5%: {r['5pct']['train_val_gap']:.2f}%")

print("\nAll plots saved to plots/")