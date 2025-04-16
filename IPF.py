import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from collections import Counter
from imblearn.under_sampling import TomekLinks  # For filtering
from imblearn.over_sampling import SMOTE  # For oversampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create an imbalanced dataset
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.9, 0.1],
                           n_informative=3, n_redundant=1, flip_y=0, 
                           n_features=10, n_clusters_per_class=1, n_samples=500, random_state=42)

# Check class distribution before filtering
print("Class distribution before filtering:", Counter(y))

# Step 1: Train a classifier (Random Forest) to find difficult samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 2: Predict instance difficulty
y_pred = clf.predict_proba(X)[:, 1]  # Get probability of being the minority class

# Define threshold: Remove samples that the model is uncertain about (borderline cases)
threshold = 0.4  # Adjust to filter more aggressively
mask = (y_pred > threshold) | (y == 1)  # Keep confident majority and all minority samples
X_filtered, y_filtered = X[mask], y[mask]

# Step 3: Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

# Check class distribution after IPF-like filtering + SMOTE
print("Class distribution after filtering & SMOTE:", Counter(y_resampled))

# Reduce dimensionality for visualization (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_resampled_pca = pca.transform(X_resampled)

# Identify synthetic samples
num_original = len(X_filtered)
synthetic_mask = np.arange(len(X_resampled)) >= num_original  # Mask for synthetic data

# Function to plot datasets with synthetic data differentiation
def plot_data(ax, X_orig, y_orig, X_syn, y_syn, title):
    ax.scatter(X_orig[y_orig == 0, 0], X_orig[y_orig == 0, 1], label="Majority Class (Original)", alpha=0.6, edgecolors='k')
    ax.scatter(X_orig[y_orig == 1, 0], X_orig[y_orig == 1, 1], label="Minority Class (Original)", alpha=0.8, edgecolors='k', marker="s")
    ax.scatter(X_syn[y_syn == 1, 0], X_syn[y_syn == 1, 1], label="Minority Class (Synthetic)", alpha=0.6, edgecolors='r', marker="x", c="red")
    ax.set_title(title)
    ax.legend()

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original dataset
plot_data(axes[0], X_pca, y, np.empty((0, 2)), np.empty(0), "Original Imbalanced Dataset")

# Filtered & Oversampled dataset
plot_data(axes[1], X_pca, y, X_resampled_pca[synthetic_mask], y_resampled[synthetic_mask], "After IPF-like Filtering + SMOTE")

plt.tight_layout()
plt.show()