import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\kiruthika\Downloads\aiml-workshop\Day3\Mall_Customers.csv")

# Convert Gender to numeric
df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

# Select relevant features for clustering
features = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

# Standardize features
x = StandardScaler().fit_transform(df[features])

# KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df["clusters"] = kmeans.fit_predict(x)

# t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
x_embedded = tsne.fit_transform(x)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=df["clusters"], cmap='tab10')
plt.title("Customer Segmentation with KMeans and t-SNE")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()
