import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

X = pd.read_csv("../Sales_Transactions_Dataset_Weekly.csv")
X = X.iloc[:, 1:53]
scaler = StandardScaler()
scaled_X = pd.DataFrame(scaler.fit_transform(X))
scaled_X.columns = X.columns

gm = GaussianMixture(n_components=3).fit(scaled_X)
y_gm = gm.predict(scaled_X)

plt.scatter(
    scaled_X[y_gm == 0]["W0"],
    scaled_X[y_gm == 0]["W2"],
    s=20,
    c="red",
    label="Cluster 1",
)
plt.scatter(
    scaled_X[y_gm == 1]["W0"],
    scaled_X[y_gm == 1]["W2"],
    s=20,
    c="green",
    label="Cluster 2",
)
plt.scatter(
    scaled_X[y_gm == 2]["W0"],
    scaled_X[y_gm == 2]["W2"],
    s=20,
    c="blue",
    label="Cluster 3",
)
plt.title("Clusters of sales")
plt.xlabel("First week")
plt.ylabel("Third week")
plt.legend()
plt.show()
