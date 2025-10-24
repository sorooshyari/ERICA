# GitHub Actions Setup for PyPI Publishing

This guide explains how to set up automated PyPI publishing using GitHub Actions.

## 🚀 Quick Setup

### 1. Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it `ERICA_PyPI` (or your preferred name)
3. Make it public (required for PyPI publishing)
4. Initialize with README (optional)

### 2. Push Your Code

```bash
# Navigate to your project directory
cd /Users/shawnshirazi/LocalExperiments/ERICA_PyPI

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: ERICA clustering package"

# Add remote origin (replace with your GitHub repo URL)
git remote add origin https://github.com/YOUR_USERNAME/ERICA_PyPI.git

# Push to GitHub
git push -u origin main
```

### 3. Configure PyPI Secrets

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `PYPI_API_TOKEN`
5. Value: Your PyPI API token (from https://pypi.org/manage/account/publishing/)

### 4. Create a Release

1. Go to your GitHub repository
2. Click **Releases** → **Create a new release**
3. Tag version: `v0.1.0`
4. Release title: `ERICA v0.1.0 - Initial Release`
5. Description: Add release notes
6. Click **Publish release**

The GitHub Action will automatically:
- Build your package
- Test it
- Upload to PyPI

## 📋 Workflow Files Created

### `.github/workflows/publish.yml`
- Triggers on new releases
- Builds and uploads to PyPI
- Can be triggered manually

### `.github/workflows/test.yml`
- Runs tests on Python 3.8-3.12
- Triggers on push/PR to main branch
- Includes coverage reporting

## 🔧 Manual Publishing

You can also trigger publishing manually:

1. Go to **Actions** tab in your GitHub repo
2. Select **Publish to PyPI** workflow
3. Click **Run workflow**
4. Select branch and click **Run workflow**

## 📦 Package Installation

After publishing, users can install with:

```bash
# Basic installation
pip install erica-clustering

# With plotting support
pip install erica-clustering[plots]

# With GUI support
pip install erica-clustering[gui]

# Full installation
pip install erica-clustering[all]
```

## 🔄 Updating the Package

To publish a new version:

1. Update version in `pyproject.toml` and `setup.py`
2. Commit changes: `git commit -m "Bump version to 0.1.1"`
3. Create new release: `v0.1.1`
4. GitHub Actions will automatically publish to PyPI

## 🛠️ Troubleshooting

### Common Issues:

1. **PyPI API Token**: Make sure it's correctly set in GitHub Secrets
2. **Package Name**: Ensure `erica-clustering` is available on PyPI
3. **Version**: Each version can only be published once
4. **Dependencies**: Check that all dependencies are available

### Check Workflow Status:

1. Go to **Actions** tab in your repository
2. Click on the workflow run to see detailed logs
3. Green checkmark = success, red X = failure

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Python Packaging User Guide](https://packaging.python.org/)
