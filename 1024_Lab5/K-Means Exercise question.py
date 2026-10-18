import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Mall_Customers.csv")

print(df.head())
print(df.isnull().sum())

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeans_Original"] = kmeans.fit_predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df["KMeans_Original"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Without Scaling")
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_scaled = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeans_Scaled"] = kmeans_scaled.fit_predict(X_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["KMeans_Scaled"])
plt.xlabel("Scaled Annual Income")
plt.ylabel("Scaled Spending Score")
plt.title("K-Means With Scaling")
plt.show()

dbscan = DBSCAN(eps=0.3, min_samples=5)
df["DBSCAN"] = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df["DBSCAN"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clustering")
plt.show()

print("K-Means Cluster Counts:")
print(df["KMeans_Scaled"].value_counts().sort_index())

print("\nDBSCAN Cluster Counts:")
print(df["DBSCAN"].value_counts().sort_index())

print("\nCluster Interpretation:")
print(df.groupby("KMeans_Scaled")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean().round(2))

inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

np.random.seed(42)
df["LuckyNumber"] = np.random.randint(0, 101, size=len(df))

X_lucky = df[
    ["Annual Income (k$)", "Spending Score (1-100)", "LuckyNumber"]
]

X_lucky_scaled = StandardScaler().fit_transform(X_lucky)

kmeans_lucky = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeans_LuckyNumber"] = kmeans_lucky.fit_predict(X_lucky_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["KMeans_LuckyNumber"]
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means With LuckyNumber")
plt.show()

print("\nCluster Counts With LuckyNumber:")
print(df["KMeans_LuckyNumber"].value_counts().sort_index())

print("\nComparison of Cluster Assignments:")
print(df[
    ["KMeans_Scaled", "KMeans_LuckyNumber"]
].head(10))