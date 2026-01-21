# 🔧 Scripts

Utility scripts for preprocessing, training, and evaluation.

## 📁 Contents

| Script | Description |
|--------|-------------|
| `preprocess_all.py` | Complete preprocessing pipeline |
| `train_unet.py` | Training script with config |
| `evaluate_unet.py` | Model evaluation and metrics |

## 🚀 Usage

### Preprocessing
```bash
python scripts/preprocess_all.py --config src/config/params.yaml
```

### Training
```bash
python scripts/train_unet.py --config src/config/params.yaml
```

### Evaluation
```bash
python scripts/evaluate_unet.py --model checkpoints/best_model.pth
```

## ⚙️ Configuration

All scripts use YAML configuration files in `src/config/`:
- `params.yaml` - Training hyperparameters
- `paths.yaml` - Data paths
- `environment.yaml` - Conda environment
