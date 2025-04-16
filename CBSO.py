import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from collections import Counter
from smote_variants import CBSO  # ✅ Correct Import for CBSO

# Create an imbalanced dataset
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.9, 0.1],
                           n_informative=3, n_redundant=1, flip_y=0, 
                           n_features=10, n_clusters_per_class=1, n_samples=500, random_state=42)

# Apply CBSO Oversampling
cbso = CBSO()
X_cbso, y_cbso = cbso.sample(X, y)

# Identify synthetic samples
num_original = len(X)
synthetic_mask = np.arange(len(X_cbso)) >= num_original  # Mask for synthetic data

# Reduce dimensionality for visualization (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_cbso_pca = pca.transform(X_cbso)

# Function to plot datasets in parallel with synthetic data differentiation
def plot_data(ax, X_orig, y_orig, X_syn, y_syn, title):
    # Plot original samples
    ax.scatter(X_orig[y_orig == 0, 0], X_orig[y_orig == 0, 1], label="Majority Class (Original)", alpha=0.6, edgecolors='k')
    ax.scatter(X_orig[y_orig == 1, 0], X_orig[y_orig == 1, 1], label="Minority Class (Original)", alpha=0.8, edgecolors='k', marker="s")

    # Plot synthetic samples (highlighted in red crosses)
    ax.scatter(X_syn[y_syn == 1, 0], X_syn[y_syn == 1, 1], label="Minority Class (Synthetic)", alpha=0.6, edgecolors='r', marker="x", c="red")
    
    ax.set_title(title)
    ax.legend()

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original dataset
plot_data(axes[0], X_pca, y, np.empty((0, 2)), np.empty(0), "Original Imbalanced Dataset")

# Oversampled dataset
plot_data(axes[1], X_pca, y, X_cbso_pca[synthetic_mask], y_cbso[synthetic_mask], "After CBSO Oversampling")

plt.tight_layout()
plt.show()