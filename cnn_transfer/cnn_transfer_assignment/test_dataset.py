from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root="data/train_data",
    transform=transform
)

print("Total images:", len(dataset))
print("Number of classes:", len(dataset.classes))
print("Classes:", dataset.classes)