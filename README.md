# GNR638: Assignment 2 — CNN Transfer Learning

Pre-trained CNN representation transfer and robustness analysis on the Aerial Images Dataset (AID), 30-class classification.

## Models
- ResNet50
- EfficientNet-B0
- ConvNeXt-Tiny

## Dataset
Download AID dataset from the course Google Drive link and place at:
```
data/train_data/
```

## Setup
```bash
pip install -r requirements.txt

# For GPU (RTX 30xx)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Run Experiments
```bash
python experiments/linear_probe.py
python experiments/fine_tune.py
python experiments/few_shot.py
python experiments/robustness.py
python experiments/feature_visualization.py
```

Plots saved to `plots/`, checkpoints saved to `models/`.

## Notes
- Windows users: use `python -m pip` instead of `pip`
- Set `num_workers=0` in DataLoader on Windows
- Seed: 42
- Add train-data in data folder- 
