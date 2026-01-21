# 🤝 Contributing to NSCLC Multi-Organ Segmentation

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)

---

## 📜 Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to:

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other contributors

---

## 🚀 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](../../issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU)

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Explain why it would benefit the project

### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 💻 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/RADIO_PROJET.git
cd RADIO_PROJET

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest black flake8 mypy
```

---

## 🔄 Pull Request Process

1. **Update documentation** if you change functionality
2. **Add tests** for new features
3. **Follow coding standards** (see below)
4. **Update CHANGELOG.md** with your changes
5. **Request review** from maintainers

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No new warnings

---

## 📝 Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints where possible
- Maximum line length: 100 characters
- Use meaningful variable names

### Documentation

- Docstrings for all public functions (Google style)
- Comments for complex logic
- Update README for new features

### Example

```python
def segment_organs(
    ct_volume: np.ndarray,
    model: nn.Module,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Segment multiple organs from a CT volume.
    
    Args:
        ct_volume: Input CT volume of shape (D, H, W)
        model: Trained U-Net model
        device: Computing device ('cuda' or 'cpu')
    
    Returns:
        Segmentation mask of shape (D, H, W) with organ labels
    
    Raises:
        ValueError: If ct_volume has invalid dimensions
    """
    # Implementation here
    pass
```

---

## 🏷️ Commit Messages

Use conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Example:
```
feat(model): add attention mechanism to U-Net decoder

- Implement self-attention blocks
- Add configuration option for attention layers
- Update documentation

Closes #42
```

---

## 📫 Questions?

Feel free to open an issue or reach out to maintainers.

Thank you for contributing! 🙏
