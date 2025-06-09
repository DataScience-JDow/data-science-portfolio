import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('data/customer_segmentation.csv')

# Basic exploration
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values (if any)
if df.isnull().sum().sum() > 0:
    df = df.dropna() # Drop rows with missing values (or use imputation if preferred)
    print("\nDropped rows with missing values. New shape:", df.shape)

# Remove unnecessary columns (e.g., CustomerID if not needed for clustering)
df = df.drop('CustoemrID', axis=1, errors='ignore') # Ignore if column doesn't exist

# Feature scaling
scaler = StandardScaler()
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
scaled_features = scaler.fit_transform(features)

# Save preprocessed data to a new CSV
preprocessed_df = pd.DataFrame(scaled_features, columns=['Age_scaled', 'Income_scaled', 'Spending_scaled'])
preprocessed_df.to_csv('data/customer_segmentation_preprocessed.csv', index=False)
print("\nPreprocessed data saved to data/customer_segmentation_preprocessed.csv")