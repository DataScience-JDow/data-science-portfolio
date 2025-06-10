# Week 2: Customer Segmentation with K-Means Clustering

## Objective
This project applies K-Means clustering to segment custoemrs based on the "Mall Customer Segmentation Data" dataset from kaggle, using features like age, annual income, and spending score.

## Methodology
- Data Source: Downloaded from Kaggle ("Mall Customer Segmentation Data").
- Preprocessing: Scaled features (Age, Annual Income, Spending Score) using StandardScaler and saved to data/customer_preprocessed.csv.
- Clustering: Implemented K-Means with 2 to 10 clusters, selecting the best number based on silhouette score (optimal clusters reported in scripts/explore_customer_segmentation.py).
- Visualization: Created a scatter plot of Income vs. Spending Score, colored by cluster, saved as plots/customer_segmentation_clusters.png.

## Results
- The optimal number of clusters was detertmined to be 6, with a silhouette score of 43.1.
- Clusters show disticnt customer segments (e.g., high-income/high-spending, low-income/low-spending).

## Next Steps
- Refine clustering with additional features or metrics if needed.
- Setup CI/CD pipeline with GitHub Actions for code linting.

### Author: DataScience-JDow Date: June 10, 2025