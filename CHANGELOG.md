# 📝 Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-01-21

### 🎉 Initial Release

#### Added
- **U-Net Architecture** for multi-organ segmentation
  - Encoder with 4 downsampling blocks
  - Skip connections for multi-scale features
  - Decoder with transposed convolutions
  - Support for 8 organ classes

- **Preprocessing Pipeline**
  - DICOM to NIfTI conversion
  - RT-STRUCT parsing and mask extraction
  - HU normalization and windowing
  - Isotropic resampling

- **Training Features**
  - Incremental training for large datasets
  - Checkpoint saving and resumption
  - Early stopping with patience
  - Mixed precision training support
  - Multi-GPU DataParallel

- **Cloud Training Support**
  - Google Colab notebooks
  - Kaggle kernel notebooks
  - Pre-configured for free GPU tiers

- **Evaluation Metrics**
  - Dice Score per organ
  - IoU (Intersection over Union)
  - Hausdorff Distance
  - Confusion matrix visualization

- **Documentation**
  - Comprehensive README
  - Installation guide
  - API documentation
  - Usage examples

#### Dataset
- Support for NSCLC-Radiomics (TCIA)
- 422 patients with CT and RT-STRUCT
- 8 anatomical structures

---

## [Unreleased]

### Planned Features
- [ ] 3D U-Net implementation
- [ ] Attention mechanisms
- [ ] nnU-Net integration
- [ ] ONNX model export
- [ ] Web demo interface
- [ ] Docker containerization

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-01-21 | Initial release |

---

## How to Update

When making changes, add entries under `[Unreleased]` section:

```markdown
### Added
- New feature description

### Changed
- Modified feature description

### Fixed
- Bug fix description

### Removed
- Removed feature description
```
