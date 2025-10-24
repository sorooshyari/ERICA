# PyPI Publishing Guide for ERICA

## Package Structure

```
ERICA_PyPI/
├── erica/                 # Main package directory
│   ├── __init__.py       # Package initialization
│   ├── core.py           # Core ERICA class
│   ├── clustering.py     # Clustering implementations
│   ├── metrics.py        # Metric calculations (CRI, WCRI, TWCRI)
│   ├── results.py        # Results handling
│   ├── utils.py          # Utility functions
│   ├── cli.py            # Command-line interface
│   └── gui.py            # Gradio GUI (optional dependency)
├── tests/                # Unit tests
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_clustering.py
│   ├── test_metrics.py
│   └── test_utils.py
├── docs/                 # Documentation
│   ├── conf.py
│   ├── index.rst
│   └── ...
├── examples/             # Usage examples
│   ├── basic_usage.py
│   ├── advanced_usage.py
│   └── notebooks/
├── setup.py              # Setup configuration
├── pyproject.toml        # Modern Python packaging
├── README.md             # Package documentation
├── LICENSE               # MIT License
├── MANIFEST.in           # Package manifest
├── requirements.txt      # Dependencies
└── .gitignore           # Git ignore rules
```

## Pre-Publishing Checklist

### 1. Complete the Core Implementation
- [ ] Implement `erica/core.py` - Main ERICA class
- [ ] Implement `erica/clustering.py` - K-Means and Agglomerative clustering
- [ ] Implement `erica/metrics.py` - CRI, WCRI, TWCRI calculations
- [ ] Implement `erica/results.py` - Results handling and export
- [ ] Implement `erica/utils.py` - Helper functions
- [ ] Implement `erica/cli.py` - Command-line interface
- [ ] Implement `erica/gui.py` - Optional Gradio interface

### 2. Write Tests
- [ ] Unit tests for core functionality
- [ ] Integration tests
- [ ] Test coverage > 80%
- [ ] All tests passing

### 3. Documentation
- [ ] Complete API documentation
- [ ] Usage examples
- [ ] Tutorial notebooks
- [ ] Contribution guidelines

### 4. Update Metadata
- [ ] Update author name and email in `setup.py` and `pyproject.toml`
- [ ] Update GitHub repository URLs
- [ ] Add proper version number
- [ ] Update LICENSE with correct year and name

### 5. Testing
```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Test CLI
erica --help
erica-gui

# Build package
python -m build

# Test installation from built package
pip install dist/erica_clustering-0.1.0-py3-none-any.whl
```

## Publishing Steps

### 1. Setup PyPI Account
1. Create account at https://pypi.org/
2. Verify email
3. Enable 2FA (recommended)
4. Create API token at https://pypi.org/manage/account/token/

### 2. Install Build Tools
```bash
pip install build twine
```

### 3. Build the Package
```bash
# Clean previous builds
rm -rf build dist *.egg-info

# Build source and wheel distributions
python -m build
```

This creates:
- `dist/erica_clustering-0.1.0.tar.gz` (source distribution)
- `dist/erica_clustering-0.1.0-py3-none-any.whl` (wheel distribution)

### 4. Test on TestPyPI (Recommended)
```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ erica-clustering
```

### 5. Publish to PyPI
```bash
# Upload to PyPI
python -m twine upload dist/*

# Enter your username (or __token__)
# Enter your password (or API token)
```

### 6. Verify Installation
```bash
pip install erica-clustering
python -c "import erica; print(erica.__version__)"
```

## Version Management

Follow Semantic Versioning (SemVer):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality
- **PATCH** version for backwards-compatible bug fixes

Example version progression:
- `0.1.0` - Initial release
- `0.1.1` - Bug fixes
- `0.2.0` - New features (backwards compatible)
- `1.0.0` - First stable release

## Continuous Integration

### GitHub Actions Workflow
Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: python -m twine upload dist/*
```

## Post-Publication

1. **Announce Release**:
   - Update GitHub repository README
   - Create release notes
   - Announce on relevant forums/communities

2. **Monitor**:
   - Check PyPI project page
   - Monitor download statistics
   - Watch for bug reports/issues

3. **Maintain**:
   - Respond to issues
   - Accept pull requests
   - Plan future releases

## Common Issues

### Issue: Package name already taken
**Solution**: Choose a different name or add a prefix/suffix

### Issue: Invalid package structure
**Solution**: Ensure proper `__init__.py` files in all package directories

### Issue: Dependencies not installing
**Solution**: Verify `install_requires` in setup.py matches requirements.txt

### Issue: Import errors after installation
**Solution**: Check package structure and `__init__.py` exports

## Resources

- **PyPI**: https://pypi.org/
- **TestPyPI**: https://test.pypi.org/
- **Packaging Guide**: https://packaging.python.org/
- **Twine Documentation**: https://twine.readthedocs.io/
- **Semantic Versioning**: https://semver.org/

## Next Steps

1. Complete the core implementation by copying relevant code from HuggingFace version
2. Write comprehensive tests
3. Create documentation
4. Test thoroughly before publishing
5. Publish to TestPyPI first
6. Once verified, publish to PyPI

