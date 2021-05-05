import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = pd.read_csv("../Sales_Transactions_Dataset_Weekly.csv")
X = X.iloc[:, 1:53]
scaler = StandardScaler()
scaled_X = pd.DataFrame(scaler.fit_transform(X))
scaled_X.columns = X.columns

wcss = []
for i in range(1, 9):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(scaled_X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 9), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(scaled_X)

plt.scatter(
    scaled_X[y_kmeans == 0]["W0"],
    scaled_X[y_kmeans == 0]["W2"],
    s=20,
    c="red",
    label="Cluster 1",
)
plt.scatter(
    scaled_X[y_kmeans == 1]["W0"],
    scaled_X[y_kmeans == 1]["W2"],
    s=20,
    c="green",
    label="Cluster 2",
)
plt.scatter(
    scaled_X[y_kmeans == 2]["W0"],
    scaled_X[y_kmeans == 2]["W2"],
    s=20,
    c="blue",
    label="Cluster 3",
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=30,
    c="yellow",
    label="Centroids",
)
plt.title("Clusters of sales")
plt.xlabel("First week")
plt.ylabel("Third week")
plt.legend()
plt.show()
