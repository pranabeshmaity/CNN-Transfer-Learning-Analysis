import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(
    root="data/train_data",
    transform=transform
)

# DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load pretrained ResNet50
model = timm.create_model(
    "resnet50",
    pretrained=True,
    num_classes=30
)

print("Model loaded successfully")
print(model)