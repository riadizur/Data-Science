import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.manifold import LocallyLinearEmbedding
from imblearn.over_sampling import SMOTE
from collections import Counter

# Create an imbalanced dataset
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.9, 0.1],
                           n_informative=3, n_redundant=1, flip_y=0, 
                           n_features=10, n_clusters_per_class=1, n_samples=500, random_state=42)

print("Class distribution before LLE-SMOTE:", Counter(y))

# Apply LLE for dimensionality reduction (before oversampling)
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=42)  # Use 2D directly for visualization
X_lle = lle.fit_transform(X)  

# Apply SMOTE after LLE transformation
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_lle, y)

print("Class distribution after LLE-SMOTE:", Counter(y_resampled))

# Identify synthetic samples
num_original = len(X)
synthetic_mask = np.arange(len(X_resampled)) >= num_original  # Mask for synthetic data

# 📌 Side-by-side plots for Before & After
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before LLE-SMOTE
axes[0].scatter(X_lle[y == 0, 0], X_lle[y == 0, 1], label="Majority Class", alpha=0.5, edgecolors='k')
axes[0].scatter(X_lle[y == 1, 0], X_lle[y == 1, 1], label="Minority Class", alpha=0.8, edgecolors='k', marker="s")
axes[0].set_title("Before LLE-SMOTE")
axes[0].legend()

# After LLE-SMOTE
axes[1].scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Majority Class", alpha=0.5, edgecolors='k')
axes[1].scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Minority Class (Original)", alpha=0.8, edgecolors='k', marker="s")
axes[1].scatter(X_resampled[synthetic_mask, 0], X_resampled[synthetic_mask, 1], label="Synthetic Minority", alpha=0.6, edgecolors='r', marker="x", c="red")
axes[1].set_title("After LLE-SMOTE")
axes[1].legend()

plt.tight_layout()
plt.show()