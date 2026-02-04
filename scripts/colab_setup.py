"""
ERICA Clustering - Google Colab Setup Script
Fixes NumPy compatibility issues and installs the library
"""

import sys
import subprocess


def fix_numpy_compatibility():
    """Fix NumPy binary incompatibility issues in Colab."""
    print("🔧 Fixing NumPy compatibility issues...")
    print("=" * 60)
    
    # Uninstall potentially conflicting packages
    print("📦 Step 1: Removing old packages...")
    packages_to_remove = ['numpy', 'pandas', 'scikit-learn']
    for package in packages_to_remove:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "uninstall", "-y", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except:
            pass
    
    # Reinstall with compatible versions
    print("📦 Step 2: Installing compatible NumPy...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--no-cache-dir", "numpy<2.0.0"
    ])
    
    print("📦 Step 3: Installing pandas and scikit-learn...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "pandas>=1.3.0", "scikit-learn>=1.0.0"
    ])
    
    print("✅ Compatibility fix complete!")
    print("=" * 60)


def install_erica():
    """Install the ERICA clustering library."""
    print("\n📦 Installing ERICA clustering library...")
    print("=" * 60)
    
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--upgrade", "erica-clustering[all]"
    ])
    
    print("✅ ERICA library installed!")
    print("=" * 60)


def verify_installation():
    """Verify that everything is working."""
    print("\n🔍 Verifying installation...")
    print("=" * 60)
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas version: {pd.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn version: {sklearn.__version__}")
        
        from erica import ERICA
        print(f"✅ ERICA imported successfully")
        
        #import gradio as gr
        #print(f"✅ Gradio version: {gr.__version__}")
        
        print("\n🎉 All packages installed and working!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        print("=" * 60)
        return False


def main():
    """Main setup function."""
    print("\n" + "=" * 60)
    print("🚀 ERICA Clustering - Colab Setup")
    print("=" * 60)
    
    try:
        # Step 1: Fix NumPy compatibility
        fix_numpy_compatibility()
        
        # Step 2: Install ERICA
        install_erica()
        
        # Step 3: Verify
        if verify_installation():
            print("\n✅ Setup complete! You can now use ERICA.")
            print("\n📝 Next steps:")
            print("   1. Restart the runtime (Runtime > Restart runtime)")
            print("   2. Run your ERICA analysis script")
            return True
        else:
            print("\n⚠️  Setup completed with warnings. Try restarting the runtime.")
            return False
            
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\n🔧 Manual fix:")
        print("   Run these commands in separate cells:")
        print("   !pip uninstall -y numpy pandas scikit-learn")
        print("   !pip install 'numpy<2.0.0' 'pandas>=1.3.0' 'scikit-learn>=1.0.0'")
        print("   !pip install erica-clustering[all]")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

