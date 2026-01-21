# 📓 Notebooks

This directory contains Jupyter notebooks for training and experimentation.

## 🌐 Cloud Training

### Google Colab
- [`colab_training.ipynb`](../colab_training.ipynb) - Basic training on Colab
- [`colab_training_multi_organ.ipynb`](../colab_training_multi_organ.ipynb) - Multi-organ segmentation
- [`colab_training_rtstruct.ipynb`](../colab_training_rtstruct.ipynb) - RT-STRUCT based training

### Kaggle
- [`kaggle_notebook_final.ipynb`](../kaggle_notebook_final.ipynb) - Production-ready Kaggle notebook
- [`kaggle_training_notebook.ipynb`](../kaggle_training_notebook.ipynb) - Standard training
- [`kaggle_training_corrected.ipynb`](../kaggle_training_corrected.ipynb) - Corrected version

## 🚀 Quick Start

1. Upload dataset to Kaggle/Colab
2. Open notebook
3. Run all cells
4. Download trained model

## ⚙️ Configuration

Modify the `CONFIG` dict in each notebook:

```python
CONFIG = {
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'patience': 10,
}
```

## 💡 Tips

- Use GPU runtime for faster training
- Save checkpoints regularly
- Monitor loss curves for overfitting
