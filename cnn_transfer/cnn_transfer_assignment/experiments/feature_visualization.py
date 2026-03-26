
import os, random
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("plots", exist_ok=True)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(root="data/train_data", transform=None)
class_names  = full_dataset.classes
NUM_CLASSES  = len(class_names)

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

class _TfDs(torch.utils.data.Dataset):
    def __init__(self, ds, indices, tf):
        self.ds = ds; self.indices = indices; self.tf = tf
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img, lbl = self.ds[self.indices[i]]; return self.tf(img), lbl

train_ds = _TfDs(full_dataset, train_idx, train_transform)
val_ds   = _TfDs(full_dataset, val_idx,   val_transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# Fixed probe subset: 30 samples per class
rng_probe = np.random.default_rng(SEED)
probe_idx = []
for label in sorted(class_to_indices):
    pool   = [i for i in val_idx if full_dataset.targets[i] == label]
    chosen = rng_probe.choice(pool, size=min(30, len(pool)), replace=False)
    probe_idx.extend(chosen.tolist())

probe_ds     = _TfDs(full_dataset, probe_idx, val_transform)
probe_loader = DataLoader(probe_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
print(f"Probe subset: {len(probe_ds)} samples ({NUM_CLASSES} classes × 30)")

# ── Per-model layer config ────────────────────────────────────────────────────
MODEL_LAYERS = {
    "resnet50": {
        "early":  "layer1",
        "middle": "layer2",
        "late1":  "layer3",
        "late2":  "layer4",
    },
    "efficientnet_b0": {
        "early":  "blocks.1",
        "middle": "blocks.3",
        "late1":  "blocks.5",
        "late2":  "blocks.6",
    },
    "convnext_tiny": {
        "early":  "stages.0",
        "middle": "stages.1",
        "late1":  "stages.2",
        "late2":  "stages.3",
    },
}

MODELS = {
    "resnet50":        "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny":   "ConvNeXt-Tiny",
}

@torch.no_grad()
def extract_features(model, loader, layer_module):
    hook_output = {}
    def hook(m, inp, out): hook_output["feat"] = out.detach()
    handle = layer_module.register_forward_hook(hook)

    feats, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        _    = model(imgs)
        f    = hook_output["feat"]
        if f.dim() == 4:
            f = f.mean(dim=[2, 3])
        elif f.dim() == 3:
            f = f.mean(dim=1)
        feats.append(f.cpu().numpy())
        labels.extend(lbls.tolist())

    handle.remove()
    return np.concatenate(feats, axis=0), np.array(labels)

def get_layer_module(model, layer_path):
    
    parts = layer_path.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

all_results = {}

for model_name, model_label in MODELS.items():
    print(f"\n{'#'*60}")
    print(f"MODEL: {model_label}")
    print(f"{'#'*60}")

    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)
    model.eval()

    layer_config = MODEL_LAYERS[model_name]
    depth_labels = list(layer_config.keys())
    probe_accs   = {}
    feature_norms = {}

    for depth, layer_path in layer_config.items():
        print(f"\n  Layer: {depth} ({layer_path})")

        layer_module = get_layer_module(model, layer_path)

        X_train, y_train = extract_features(model, train_loader, layer_module)
        X_val,   y_val   = extract_features(model, val_loader,   layer_module)

        norms = np.linalg.norm(X_val, axis=1)
        feature_norms[depth] = {"mean": float(norms.mean()), "std": float(norms.std())}
        print(f"    Feature dim: {X_train.shape[1]} | Norm: {norms.mean():.4f} ± {norms.std():.4f}")

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_v_s  = scaler.transform(X_val)

        n_comp = min(128, X_tr_s.shape[0]-1, X_tr_s.shape[1])
        pca_r  = PCA(n_components=n_comp, random_state=SEED)
        X_tr_s = pca_r.fit_transform(X_tr_s)
        X_v_s  = pca_r.transform(X_v_s)

        clf = SGDClassifier(loss="modified_huber", max_iter=1000,
                            random_state=SEED, tol=1e-3)
        clf.fit(X_tr_s, y_train)
        acc = float(clf.score(X_v_s, y_val)) * 100
        probe_accs[depth] = acc
        print(f"    Linear probe val acc: {acc:.2f}%")

        # PCA 2D plot on probe subset
        X_probe, y_probe = extract_features(model, probe_loader, layer_module)
        reduced = PCA(n_components=2, random_state=SEED).fit_transform(
            StandardScaler().fit_transform(X_probe))

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                              c=y_probe, cmap="tab20", s=12, alpha=0.7)
        plt.colorbar(scatter, label="Class index")
        plt.title(f"{model_label} – {depth} layer PCA")
        plt.tight_layout()
        plt.savefig(f"plots/{model_name}_probe_{depth}_pca.png", dpi=150)
        plt.close()

    all_results[model_name] = {
        "probe_accs":    probe_accs,
        "feature_norms": feature_norms,
    }

    # Plot: Val accuracy vs depth
    accs_ordered = [probe_accs[d] for d in depth_labels]
    plt.figure(figsize=(8, 5))
    plt.plot(depth_labels, accs_ordered, "o-", linewidth=2, markersize=8, color="tab:blue")
    for x, y in zip(depth_labels, accs_ordered):
        plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(5,5), fontsize=9)
    plt.xlabel("Layer Depth"); plt.ylabel("Linear Probe Val Accuracy (%)")
    plt.title(f"{model_label} – Accuracy vs Layer Depth")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_probe_accuracy_depth.png", dpi=150)
    plt.close()

    # Plot: Feature norm vs depth
    norm_means = [feature_norms[d]["mean"] for d in depth_labels]
    plt.figure(figsize=(7, 4))
    plt.bar(depth_labels, norm_means, color="tab:orange")
    plt.ylabel("Mean Feature Norm (L2)")
    plt.title(f"{model_label} – Feature Norm vs Depth")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_feature_norms_depth.png", dpi=150)
    plt.close()

    print(f"\nSummary for {model_label}:")
    for d in depth_labels:
        print(f"  {d:<8} acc={probe_accs[d]:>6.2f}%  "
              f"norm={feature_norms[d]['mean']:.4f} ± {feature_norms[d]['std']:.4f}")


print(f"\n{'='*60}")
print("LAYER PROBING FULL SUMMARY")
print(f"{'='*60}")
for model_name, model_label in MODELS.items():
    print(f"\n{model_label}:")
    r = all_results[model_name]
    for d in ["early", "middle", "late1", "late2"]:
        print(f"  {d:<8} acc={r['probe_accs'][d]:>6.2f}%  "
              f"norm={r['feature_norms'][d]['mean']:.4f}")

print("\nAll plots saved to plots/")