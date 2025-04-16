import smogn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Step 1: Generate Regression Dataset (500 samples, 5 features)
np.random.seed(42)
X = np.random.rand(500, 5)
y = np.concatenate([np.random.rand(350) * 10, np.random.rand(150) * 100])  # Imbalanced target

# ✅ Step 2: Introduce 30% Missing Values in Target
missing_indices = np.random.choice(range(500), size=int(0.3 * 500), replace=False)
y[missing_indices] = np.nan  # Assign NaN to 30% of target values

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y  # ✅ Keep missing values in original dataset

# ✅ Step 3: Create a Modified Copy (Replacing NaNs with 0 for SMOGN Processing)
df_for_smogn = df.copy()
df_for_smogn['target'].fillna(0, inplace=True)  # Replace NaNs with zero only for processing

# ✅ Step 4: Apply SMOGN on the modified dataset
df_resampled = smogn.smoter(data=df_for_smogn, y='target')

# ✅ Step 5: Restore the Original Target Values for Existing Data
df_resampled = df_resampled.merge(df[['target']], left_index=True, right_index=True, how="left")
df_resampled['target_x'] = df_resampled['target_y'].combine_first(df_resampled['target_x'])
df_resampled.drop(columns=['target_y'], inplace=True)
df_resampled.rename(columns={'target_x': 'target'}, inplace=True)

# ✅ Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df_resampled.drop(columns=['target']), 
                                                    df_resampled['target'], test_size=0.2, random_state=42)

# ✅ Step 7: Train Regression Model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Step 8: Make Predictions & Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ✅ Step 9: Create Before & After Tables
df_before = df.copy()
df_before['target'].fillna(0, inplace=True)  # Show NaNs as zero in the original table

df_after = df_resampled.copy()

# ✅ Step 10: Display Target Distribution Before & After SMOGN
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_before['target'], bins=30, color='blue', kde=True)
plt.title("Original Target Distribution (Before SMOGN)")

plt.subplot(1, 2, 2)
sns.histplot(df_after['target'], bins=30, color='green', kde=True)
plt.title("Resampled Target Distribution (After SMOGN)")

plt.show()

# ✅ Step 11: Print Model Performance
print(f"Model Performance after SMOGN Oversampling:")
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# ✅ Step 12: Display Before & After Tables (First 10 Rows)
print("\n📌 First 10 Rows - Original Data (Before SMOGN, NaN Replaced with 0):")
print(df_before.head(10))

print("\n📌 First 10 Rows - After SMOGN (Only NaN Values are Synthetic):")
print(df_after.head(10))