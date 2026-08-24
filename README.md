# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries
2.Load the dataset (Mall_Customers.csv)
3.Check dataset info,Check for missing values
4.Select features: Annual Income and Spending Score
5.Standardize the selected features
6.Apply Elbow method for k = 1 to 10 and Plot
7.Apply Silhouette Score for k = 2 to 10 and Plot
8.Fit K-Means model
9.Predict cluster labels and add cluster labels to the dataset 10.Compute cluster centers
10.Visualize clusters using scatter plot

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------
# 1. Load the dataset
# ---------------------------------------
df = pd.read_csv("Mall_Customers.csv")  # UPDATE PATH IF NEEDED
print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
display(df.head())

# ---------------------------------------
# 2. Check info and missing values
# ---------------------------------------
print("\nDataset Info:")
display(df.info())
print("\nMissing Values:")
display(df.isnull().sum())

# ---------------------------------------
# 3. Select features for clustering
# Using Income & Spending Score
# ---------------------------------------
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

print("\nFeatures Used:", features)

# ---------------------------------------
# 4. Standardize the data
# ---------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------
# 5. Elbow Method to choose k
# ---------------------------------------
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia / SSE")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# ---------------------------------------
# 6. Silhouette Scores
# ---------------------------------------
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(6,4))
plt.plot(range(2, 11), sil_scores, marker='o', color="orange")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.grid(True)
plt.show()

# ---------------------------------------
# 7. Apply KMeans (Choose k=5 by elbow)
# ---------------------------------------
k_final = 5
kmeans = KMeans(n_clusters=k_final, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_labels
print("\nCluster Counts:")
print(df["Cluster"].value_counts())

# ---------------------------------------
# 8. Cluster Centers in original units
# ---------------------------------------
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(centers_original, columns=features)
centers_df["Cluster"] = range(k_final)

print("\nCluster Centers (Original Values):")
display(centers_df.round(2))

# ---------------------------------------
# 9. Visualization of Clusters
# ---------------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10",
    s=70
)

# Show cluster centers
plt.scatter(
    centers_df["Annual Income (k$)"],
    centers_df["Spending Score (1-100)"],
    s=250,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means (k=5)")
plt.legend()
plt.grid(True)
plt.show()



/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SANJAI K
RegisterNumber:  212225230245
*/
```

## Output:
<img width="848" height="410" alt="image" src="https://github.com/user-attachments/assets/59d50412-4870-4026-9518-3c4b8bc5e291" />
<img width="1198" height="403" alt="image" src="https://github.com/user-attachments/assets/b92e9c96-c1b9-4d25-805c-649d85921d70" />
<img width="950" height="416" alt="image" src="https://github.com/user-attachments/assets/d41fa841-125e-4515-840a-db20d7fc3d28" />
<img width="996" height="420" alt="image" src="https://github.com/user-attachments/assets/9f8c3502-603b-47ee-85ef-3ee8f6a4b538" />
<img width="1181" height="361" alt="image" src="https://github.com/user-attachments/assets/808959a0-bd47-45d1-8375-c979b16bda92" />
<img width="1005" height="401" alt="image" src="https://github.com/user-attachments/assets/02762565-30d8-4bae-b7a9-3c1f283b282d" />

<img width="991" height="395" alt="image" src="https://github.com/user-attachments/assets/10301e90-cd18-4e1a-917a-25dcc67ba141" />





## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
