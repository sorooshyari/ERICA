# ERICA PyPI Package - TODO List

## Phase 1: Core Implementation (NEXT STEPS)

### High Priority
- [ ] Create `erica/core.py` - Main ERICA class
  - Port core functionality from HuggingFace version
  - Implement clean Python API
  - Add proper docstrings and type hints
  
- [ ] Create `erica/clustering.py` - Clustering implementations
  - K-Means wrapper with ERICA logic
  - Agglomerative clustering wrapper
  - Iterative subsampling logic
  
- [ ] Create `erica/metrics.py` - Metrics calculations
  - CRI computation
  - WCRI computation  
  - TWCRI computation
  - Helper functions for metric calculations
  
- [ ] Create `erica/results.py` - Results handling
  - ERICAResults class
  - Save/load functionality
  - Plotting methods
  - Export to various formats

### Medium Priority
- [ ] Create `erica/utils.py` - Utility functions
  - Data loading (CSV, NPY)
  - Data validation
  - Progress tracking
  - Logging utilities
  
- [ ] Create `erica/cli.py` - Command-line interface
  - `erica analyze` command
  - `erica metrics` command
  - `erica plot` command
  - Argument parsing and validation
  
#- [ ] Create `erica/gui.py` - Optional Gradio interface
#  - Port from HuggingFace version
#  - Simplify for package usage
#  - Make it optional dependency

## Phase 2: Testing

- [ ] Create `tests/test_core.py`
  - Test ERICA class initialization
  - Test fit method
  - Test error handling
  
- [ ] Create `tests/test_clustering.py`
  - Test K-Means clustering
  - Test Agglomerative clustering
  - Test subsampling logic
  
- [ ] Create `tests/test_metrics.py`
  - Test CRI calculation
  - Test WCRI calculation
  - Test TWCRI calculation
  
- [ ] Create `tests/test_utils.py`
  - Test data loading
  - Test validation
  
- [ ] Setup pytest configuration
- [ ] Setup coverage reporting
- [ ] Aim for >80% test coverage

## Phase 3: Documentation

- [ ] Write comprehensive docstrings for all public APIs
- [ ] Create API documentation with Sphinx
- [ ] Write usage tutorials
- [ ] Create example notebooks
  - Basic usage example
  - Advanced configuration example
  - Comparison of methods example
  
- [ ] Write contribution guidelines
- [ ] Create changelog

## Phase 4: Examples

- [ ] Create `examples/basic_usage.py`
- [ ] Create `examples/advanced_usage.py`
- [ ] Create `examples/comparing_methods.py`
- [ ] Create `examples/custom_visualization.py`
- [ ] Create Jupyter notebooks in `examples/notebooks/`

## Phase 5: Packaging & Distribution

### Pre-Publication
- [ ] Update author information in all files
- [ ] Update GitHub URLs (once repository is created)
- [ ] Verify all dependencies
- [ ] Test installation in clean environment
- [ ] Run all tests
- [ ] Build documentation
- [ ] Review README
- [ ] Review LICENSE

### Testing
- [ ] Test build process: `python -m build`
- [ ] Test local installation: `pip install -e .`
- [ ] Test CLI commands
- [ ] Test GUI (if applicable)
- [ ] Upload to TestPyPI
- [ ] Test installation from TestPyPI
- [ ] Fix any issues found

### Publication
- [ ] Create PyPI account (if needed)
- [ ] Generate API token
- [ ] Build final distributions
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Verify installation: `pip install erica-clustering`
- [ ] Test installed package

## Phase 6: Post-Publication

- [ ] Create GitHub repository
- [ ] Push code to GitHub
- [ ] Create first release on GitHub
- [ ] Setup GitHub Actions for CI/CD
- [ ] Setup ReadTheDocs for documentation
- [ ] Announce release
- [ ] Monitor for issues
- [ ] Plan next version features

## Optional Enhancements

- [ ] Add support for more clustering algorithms
- [ ] Implement parallel processing optimization
- [ ] Add more visualization options
- [ ] Create Docker image
- [ ] Add support for streaming large datasets
- [ ] Implement caching for repeated analyses
- [ ] Add configuration file support
- [ ] Create web-based demo

## Questions to Resolve

1. **Package name**: Confirm "erica-clustering" is available on PyPI
2. **Author information**: Update with correct name and email
3. **Repository location**: Where will the GitHub repo be hosted?
4. **License**: Confirm MIT license is appropriate
5. **Version strategy**: Confirm starting with 0.1.0
6. **Documentation hosting**: ReadTheDocs vs GitHub Pages vs both?
7. **CI/CD**: GitHub Actions, Travis CI, or CircleCI?

## Current Status

✅ **COMPLETED**:
- Project structure created
- Setup files configured (setup.py, pyproject.toml)
- README.md written
- LICENSE file created
- Requirements specified
- .gitignore configured
- PyPI publishing guide created

🚧 **IN PROGRESS**:
- Core implementation (starting next)

⏳ **NOT STARTED**:
- Testing
- Documentation
- Examples
- Publication

