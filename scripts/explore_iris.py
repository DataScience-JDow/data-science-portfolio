import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/iris.csv')

print("First 5 rows of the dataset:")
print(df.head(5))
print("\nSummary satatistics:")
print(df.describe())

# Visualize: Scatter plot of sepal dimensions by species
plt.figure(figsize=(8,6))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df)
plt.title('Iris Sepal Dimensions by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.savefig('plots/iris_sepal_scatter.png')
plt.show()