#!/usr/bin/env python3
"""
Script to bump version numbers in pyproject.toml and setup.py
"""

import re
import sys
from pathlib import Path


def bump_version(current_version, bump_type):
    """Bump version number based on type."""
    major, minor, patch = map(int, current_version.split('.'))
    
    if bump_type == 'major':
        return f"{major + 1}.0.0"
    elif bump_type == 'minor':
        return f"{major}.{minor + 1}.0"
    elif bump_type == 'patch':
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError("bump_type must be 'major', 'minor', or 'patch'")


def update_pyproject_toml(version):
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    # Replace version line
    content = re.sub(
        r'version = "[^"]*"',
        f'version = "{version}"',
        content
    )
    
    pyproject_path.write_text(content)
    print(f"Updated pyproject.toml to version {version}")


def update_setup_py(version):
    """Update version in setup.py."""
    setup_path = Path("setup.py")
    content = setup_path.read_text()
    
    # Replace version line
    content = re.sub(
        r"version='[^']*'",
        f"version='{version}'",
        content
    )
    
    setup_path.write_text(content)
    print(f"Updated setup.py to version {version}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python bump_version.py <major|minor|patch>")
        sys.exit(1)
    
    bump_type = sys.argv[1]
    
    if bump_type not in ['major', 'minor', 'patch']:
        print("Error: bump_type must be 'major', 'minor', or 'patch'")
        sys.exit(1)
    
    # Read current version from pyproject.toml
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    match = re.search(r'version = "([^"]*)"', content)
    
    if not match:
        print("Error: Could not find version in pyproject.toml")
        sys.exit(1)
    
    current_version = match.group(1)
    new_version = bump_version(current_version, bump_type)
    
    print(f"Bumping version from {current_version} to {new_version}")
    
    # Update files
    update_pyproject_toml(new_version)
    update_setup_py(new_version)
    
    print(f"\nVersion bumped to {new_version}")
    print("Next steps:")
    print("1. git add .")
    print("2. git commit -m 'Bump version to {new_version}'")
    print("3. git tag v{new_version}")
    print("4. git push origin main --tags")
    print("5. Create a new release on GitHub")


if __name__ == "__main__":
    main()
