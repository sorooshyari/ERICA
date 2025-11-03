#!/usr/bin/env python
"""
Quick test script to verify the demo interface imports and basic functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import gradio as gr
        print(f"✓ Gradio {gr.__version__}")
    except ImportError as e:
        print(f"✗ Gradio import failed: {e}")
        return False
    
    try:
        from erica import ERICA
        print("✓ ERICA import successful")
    except ImportError as e:
        print(f"✗ ERICA import failed: {e}")
        return False
    
    try:
        from erica.data import load_data, get_dataset_info
        print("✓ Data utilities import successful")
    except ImportError as e:
        print(f"✗ Data utilities import failed: {e}")
        return False
    
    try:
        from erica.plotting import (
            plot_metrics,
            plot_clam_heatmap,
            plot_k_star_selection,
            plot_k_star_by_method,
        )
        print("✓ Plotting utilities import successful")
    except ImportError as e:
        print(f"✗ Plotting utilities import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from erica.data import get_dataset_info
        
        # Create test data
        test_data = np.random.rand(50, 20)
        info = get_dataset_info(test_data, transpose=False)
        
        assert info['n_samples'] == 50
        assert info['n_features'] == 20
        print("✓ Basic data processing works")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ERICA Demo Test Script")
    print("=" * 60)
    
    imports_ok = test_imports()
    if not imports_ok:
        print("\n❌ Import tests failed. Please install missing dependencies.")
        sys.exit(1)
    
    func_ok = test_basic_functionality()
    if not func_ok:
        print("\n❌ Functionality tests failed.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! You can run the demo with:")
    print("  python demo/gradio_demo.py")
    print("=" * 60)

