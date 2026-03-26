"""
Scenario 4.2 – Fine-Tuning Strategies

"""

import os, random
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("plots", exist_ok=True)

# ── Transforms ────────────────────────────────────────────────────────────────
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

# ──split ──────────────────────────────────────────────────────────
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

train_dataset = _SubsetWithTransform(full_dataset, train_idx, train_transform)
val_dataset   = _SubsetWithTransform(full_dataset, val_idx,   val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# ── Models to run ─────────────────────────────────────────────────────────────
MODELS = {
    "resnet50":        "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny":   "ConvNeXt-Tiny",
}

# ── Per-model last block config ───────────────────────────────────────────────
LAST_BLOCK = {
    "resnet50":        "layer4",
    "efficientnet_b0": "blocks",
    "convnext_tiny":   "stages",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total

# ── Strategy runner ───────────────────────────────────────────────────────────
def train_strategy(strategy_name, model_name, epochs=7):

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))

    if strategy_name == "linear_probe":
        for p in model.parameters(): p.requires_grad = False
        for p in model.get_classifier().parameters(): p.requires_grad = True

    elif strategy_name == "last_block":
        for p in model.parameters(): p.requires_grad = False
        # Unfreeze last block depending on architecture
        last_block_name = LAST_BLOCK[model_name]
        last_block = getattr(model, last_block_name, None)
        if last_block is not None:
            # For resnet50: layer4 is directly usable
            # For efficientnet/convnext: unfreeze only the last sub-block
            if hasattr(last_block, '__iter__'):
                for p in list(last_block)[-1].parameters(): p.requires_grad = True
            else:
                for p in last_block.parameters(): p.requires_grad = True
        for p in model.get_classifier().parameters(): p.requires_grad = True

    elif strategy_name == "full_finetune":
        for p in model.parameters(): p.requires_grad = True

    elif strategy_name == "selective_20pct":
        for p in model.parameters(): p.requires_grad = False
        total_params = sum(p.numel() for p in model.parameters())
        budget       = int(total_params * 0.20)
        unfrozen     = 0
        for name, param in reversed(list(model.named_parameters())):
            if unfrozen >= budget: break
            param.requires_grad = True
            unfrozen += param.numel()
        actual_pct = unfrozen / total_params
        print(f"  [selective_20pct] unfrozen {unfrozen:,}/{total_params:,} = {actual_pct:.1%}")

    model = model.to(device)

    total, trainable = count_params(model)
    trainable_frac   = trainable / total
    print(f"\n{'='*50}\nStrategy: {strategy_name}")
    print(f"  Total: {total:,} | Trainable: {trainable:,} ({trainable_frac:.2%})")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4 if strategy_name == "full_finetune" else 1e-3,
        weight_decay=1e-4)

    loss_curve = []
    grad_norms = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if epoch == epochs - 1:
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        grad_norms[name] = p.grad.norm().item()

            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        avg_loss  = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        loss_curve.append(avg_loss)
        print(f"  Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

    acc = evaluate(model)
    print(f"  Val Accuracy: {acc:.2f}%")

    return {
        "acc":            acc,
        "trainable_frac": trainable_frac,
        "loss_curve":     loss_curve,
        "grad_norms":     grad_norms,
    }


# ── Run all models ────────────────────────────────────────────────────────────
strategies  = ["linear_probe", "last_block", "selective_20pct"]
all_results = {}

SKIP_MODELS = ["resnet50", "efficientnet_b0"]

for model_name, model_label in MODELS.items():

    

    print(f"\n{'#'*60}")
    print(f"MODEL: {model_label}")
    print(f"{'#'*60}")

    results = {}
    for s in strategies:
        epochs_map = {
    "linear_probe":    5,
    "last_block":      5,
    "selective_20pct": 5,
    "full_finetune":   3,
}
    results[s] = train_strategy(s, model_name, epochs=epochs_map[s])

    all_results[model_name] = results

    # ── Plot 1: Accuracy vs % unfrozen params ─────────────────────────────────
    fracs = [results[s]["trainable_frac"] * 100 for s in strategies]
    accs  = [results[s]["acc"]            for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    for s, f, a in zip(strategies, fracs, accs):
        ax.scatter(f, a, s=120, zorder=3, label=s)
        ax.annotate(f" {s}", (f, a), fontsize=8)
    ax.set_xlabel("Trainable Parameters (%)")
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title(f"{model_label} – Accuracy vs % Unfrozen Parameters")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_acc_vs_unfrozen.png", dpi=150)
    plt.close()

    # ── Plot 2: Bar chart comparison ──────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    bars = plt.bar(strategies, accs, color=plt.cm.Set2.colors[:4])
    for bar, v in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", fontsize=9)
    plt.ylabel("Val Accuracy (%)")
    plt.title(f"{model_label} – Fine-Tuning Strategy Comparison")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_fine_tuning_comparison.png", dpi=150)
    plt.close()

    # ── Plot 3: Convergence curves ────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for s, c in zip(strategies, colours):
        plt.plot(results[s]["loss_curve"], label=s, color=c)
    plt.xlabel("Epoch"); plt.ylabel("Train Loss")
    plt.title(f"{model_label} – Convergence Stability")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_convergence.png", dpi=150)
    plt.close()

    # ── Plot 4: Gradient norms ────────────────────────────────────────────────
    gn     = results["full_finetune"]["grad_norms"]
    names  = [n for n in gn if "weight" in n][:40]
    values = [gn[n] for n in names]

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(names)), values, color="steelblue")
    plt.xticks(range(len(names)), [n.replace(".weight", "") for n in names],
               rotation=75, ha="right", fontsize=6)
    plt.ylabel("Gradient Norm")
    plt.title(f"{model_label} – Gradient Norms (full_finetune, final epoch)")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_gradient_norms.png", dpi=150)
    plt.close()

    # ── Print summary per model ───────────────────────────────────────────────
    print(f"\nSummary for {model_label}:")
    for s in strategies:
        print(f"  {s:<20} | Trainable: {results[s]['trainable_frac']:.1%} | "
              f"Val Acc: {results[s]['acc']:.2f}%")


print(f"\n{'='*60}")
print("FINE-TUNING FULL SUMMARY")
print(f"{'='*60}")
for model_name, model_label in MODELS.items():
    print(f"\n{model_label}:")
    for s in strategies:
        r = all_results[model_name][s]
        print(f"  {s:<20} | Trainable: {r['trainable_frac']:.1%} | Val Acc: {r['acc']:.2f}%")

print("\nAll plots saved to plots/")