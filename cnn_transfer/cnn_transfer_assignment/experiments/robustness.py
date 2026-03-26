"""
Scenario 4.4 – Corruption Robustness Evaluation

"""

import os, random
import torch
import torch.nn.functional as F
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# ── Corruption functions ──────────────────────────────────────────────────────
def gaussian_noise(img, sigma):
    return torch.clamp(img + torch.randn_like(img) * sigma, 0, 1)

def motion_blur(img, size=15):
    kernel = torch.ones(1, 1, 1, size) / size
    kernel = kernel.expand(img.shape[0], 1, 1, size)
    blurred = F.conv2d(img.unsqueeze(0), kernel, padding=(0, size//2), groups=img.shape[0])
    return blurred.squeeze(0).clamp(0, 1)

def brightness_shift(img, delta=0.5):
    return torch.clamp(img + delta, 0, 1)

# ── Dataset with corruption ───────────────────────────────────────────────────
class CorruptedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, corruption_fn=None):
        self.base_ds    = base_ds
        self.corruption = corruption_fn
        self.to_tensor  = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self): return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        t = self.to_tensor(img)
        if self.corruption is not None:
            t = self.corruption(t)
        t = normalize(t)
        return t, label

# ── Dataset setup ─────────────────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(root="data/train_data", transform=None)

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

val_base = torch.utils.data.Subset(full_dataset, val_idx)

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "resnet50":        "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny":   "ConvNeXt-Tiny",
}

def evaluate(model, corruption_fn=None):
    ds     = CorruptedDataset(val_base, corruption_fn)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
    return 100 * correct / total

all_results = {}

for model_name, model_label in MODELS.items():
    print(f"\n{'#'*60}")
    print(f"MODEL: {model_label}")
    print(f"{'#'*60}")

    model = timm.create_model(model_name, pretrained=True,
                               num_classes=len(full_dataset.classes))
    for p in model.parameters(): p.requires_grad = False
    for p in model.get_classifier().parameters(): p.requires_grad = True
    model = model.to(device)

    ckpt_path = f"models/{model_name}_linear_probe.pth"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path} …")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"No checkpoint found — training linear probe …")
        train_transform = transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(), normalize])

        class _TfDs(torch.utils.data.Dataset):
            def __init__(self, ds, tf): self.ds = ds; self.tf = tf
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                img, lbl = self.ds[i]; return self.tf(img), lbl

        train_ds     = torch.utils.data.Subset(full_dataset, train_idx)
        train_loader = DataLoader(_TfDs(train_ds, train_transform),
                                  batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer  = optim.Adam(model.get_classifier().parameters(), lr=1e-3)

        for epoch in range(10):
            model.train()
            correct = total = 0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                criterion(outputs, lbls).backward()
                optimizer.step()
                correct += (outputs.argmax(1) == lbls).sum().item()
                total   += lbls.size(0)
            print(f"  Epoch {epoch+1:02d}/10 | Train Acc: {100*correct/total:.2f}%")

        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    # Evaluate corruptions
    print(f"\nEvaluating corruptions …")
    clean_acc = evaluate(model, None)
    print(f"Clean accuracy: {clean_acc:.2f}%")

    corruption_suite = {
        "Clean":            None,
        "Gaussian_s005":    lambda x: gaussian_noise(x, 0.05),
        "Gaussian_s010":    lambda x: gaussian_noise(x, 0.10),
        "Gaussian_s020":    lambda x: gaussian_noise(x, 0.20),
        "Motion_Blur":      motion_blur,
        "Brightness_Shift": brightness_shift,
    }

    results = {}
    for name, fn in corruption_suite.items():
        acc = clean_acc if fn is None else evaluate(model, fn)
        ce  = 1 - acc / 100.0
        rr  = acc / clean_acc
        results[name] = {"acc": acc, "corruption_error": ce, "relative_robustness": rr}
        print(f"  {name:<22} Acc={acc:>6.2f}%  CE={ce:.4f}  RR={rr:.4f}")

    all_results[model_name] = results

    # Plot: Accuracy bar chart
    names   = list(results.keys())
    accs    = [results[n]["acc"] for n in names]
    colours = ["green"] + ["steelblue"]*3 + ["darkorange", "purple"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, accs, color=colours)
    for bar, v in zip(bars, accs):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{v:.1f}%", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right"); plt.ylabel("Accuracy (%)")
    plt.title(f"{model_label} – Robustness to Corruption")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_robustness_results.png", dpi=150)
    plt.close()

    # Plot: Relative robustness
    rr_vals = [results[n]["relative_robustness"] for n in names]
    plt.figure(figsize=(10, 5))
    plt.bar(names, rr_vals, color=colours)
    plt.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Clean baseline")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Relative Robustness")
    plt.title(f"{model_label} – Relative Robustness")
    plt.grid(axis="y", alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_relative_robustness.png", dpi=150)
    plt.close()


print(f"\n{'='*60}")
print("ROBUSTNESS FULL SUMMARY")
print(f"{'='*60}")
for model_name, model_label in MODELS.items():
    print(f"\n{model_label}:")
    for name, r in all_results[model_name].items():
        print(f"  {name:<22} Acc={r['acc']:>6.2f}%  CE={r['corruption_error']:.4f}  RR={r['relative_robustness']:.4f}")

print("\nAll plots saved to plots/")