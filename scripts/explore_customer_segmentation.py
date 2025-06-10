import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# K-Means Clustering
range_n_clusters = range(2,11) # Test 2 to 10 clusters
best_score = -1
best_n_clusters = 2

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.3f}")
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_n_clusters = n_clusters

print(f"\nBest number of clusters: {best_n_clusters} with Silhouette Score: {best_score:.3f}")

# Fit final K-Means with the best number of clusters
final_kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
final_clusters = final_kmeans.fit_predict(scaled_features)

# Add cluster labels to the preprocessed DataFrame
preprocessed_df['Cluster'] = final_clusters
preprocessed_df.to_csv('data/customer_segmentation_preprocessed.csv', index=False)
print("Updated preprocessed data with cluster labels saved.")

# Visualization
plt.figure(figsize=(10,6))
sns.scatterplot(x='Income_scaled', y='Spending_scaled', hue='Cluster', data=preprocessed_df, palette='deep', s=100)
plt.title('Customer Segments Based on Income and Spending Score')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.legend(title='Cluster')
plt.savefig('plots/cluster_segmentation_clusters.png')
plt.show()