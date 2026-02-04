# 🚀 ERICA Clustering - Google Colab Instructions

## Simple Installation (Recommended)

ERICA v0.1.3+ is fully compatible with modern NumPy 2.x, pandas 2.x, and scikit-learn 1.5+!

### ✅ Quick Start - Single Cell Installation:

```python
# Install ERICA clustering library with all features
!pip install --upgrade erica-clustering[all]

# Verify installation
import numpy as np
import pandas as pd
from erica import ERICA

print(f"✅ NumPy version: {np.__version__}")
print(f"✅ Pandas version: {pd.__version__}")
print("✅ ERICA ready to use!")
```

That's it! No special fixes needed with version 0.1.3+.

---

## Legacy Fix (Only for v0.1.2 or earlier)

If you're using ERICA v0.1.2 or earlier and see: `ValueError: numpy.dtype size changed, may indicate binary incompatibility`

### Run these cells in order:

#### Cell 1: Upgrade to Latest Version
```python
# Upgrade to the latest ERICA version (recommended)
!pip install --upgrade erica-clustering[all]
```

#### Cell 3: Restart Runtime
```python
# IMPORTANT: Restart the runtime now!
# Go to: Runtime > Restart runtime
# Then continue with the cells below
```

#### Cell 4: Verify Installation
```python
# Verify everything is working
import numpy as np
import pandas as pd
from erica import ERICA

print(f"✅ NumPy version: {np.__version__}")
print(f"✅ Pandas version: {pd.__version__}")
print(f"✅ ERICA imported successfully!")
```

#### Cell 5: Run ERICA Gradio App
```python
# Download and run the Gradio app
!wget https://raw.githubusercontent.com/astrocyte/ERICA_clustering/main/erica_gradio_app.py
!python erica_gradio_app.py
```

---

## 📦 Alternative: One-Line Auto-Fix

If you want an automated fix, run this:

```python
!pip install --upgrade --force-reinstall 'numpy<2.0.0' pandas scikit-learn
!pip install erica-clustering[all]

# Then restart runtime: Runtime > Restart runtime
```

---

## 🎯 Quick Example Without Gradio

If you just want to use ERICA directly in Colab:

```python
# After fixing NumPy compatibility above...
import numpy as np
from erica import ERICA

# Generate sample data
data = np.random.rand(100, 50)

# Run ERICA analysis
erica = ERICA(
    data=data,
    k_range=[2, 3, 4, 5],
    n_iterations=200,
    method='kmeans'
)

results = erica.run()

# Get metrics
metrics = erica.get_metrics()

# Display results
for k in [2, 3, 4, 5]:
    cri = metrics[k]['kmeans']['CRI']
    wcri = metrics[k]['kmeans']['WCRI']
    twcri = metrics[k]['kmeans']['TWCRI']
    print(f"K={k}: CRI={cri:.3f}, WCRI={wcri:.3f}, TWCRI={twcri:.3f}")

# Plot results
fig1, fig2 = erica.plot_metrics()
fig1.show()
```

---

## 🔍 Why This Happens

Google Colab pre-installs certain packages that may be compiled against different NumPy versions. When you install new packages, they may expect a different NumPy ABI (Application Binary Interface).

**The fix:** Reinstall NumPy and dependent packages with compatible versions.

---

## 💡 Pro Tips

1. **Always restart runtime** after fixing NumPy compatibility
2. **Use `numpy<2.0.0`** for maximum compatibility with existing packages
3. **Install ERICA after** fixing NumPy issues
4. **Save your fixes** in a setup cell at the top of your notebook

---

## 🆘 Still Having Issues?

If the above doesn't work, try this nuclear option:

```python
# Reset everything
!pip uninstall -y numpy pandas scikit-learn matplotlib plotly gradio

# Fresh install with specific versions
!pip install 'numpy==1.26.4' 'pandas==2.2.3' 'scikit-learn==1.5.0'
!pip install 'matplotlib>=3.5.0' 'plotly>=5.0.0'
!pip install 'gradio>=4.0.0'

# Then install ERICA
!pip install erica-clustering

# Restart runtime: Runtime > Restart runtime
```

---

## 📚 Additional Resources

- **PyPI Package**: https://pypi.org/project/erica-clustering/
- **GitHub**: https://github.com/astrocyte/ERICA_clustering
- **Documentation**: See README.md for full API reference

---

**Need Help?** Open an issue on GitHub with your error message and Colab environment details.

