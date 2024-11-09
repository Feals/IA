import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

np.random.seed(808)

centers = [[2, 2], [-2, -2], [2, -2]]

X, labels_true = make_blobs(
                      n_samples=3000, 
                      centers=centers, 
                      cluster_std=0.7)

n_clusters = len(centers)

fig = plt.figure(figsize=(6, 6))
colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

ax = fig.add_subplot(1, 1, 1)

ax.set_title("3 clusters")

for k, col in zip(range(len(centers)), colors):
    my_members = labels_true == k
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker="o", markersize=4, alpha = 1)

for k, col in zip(range(len(centers)), colors):
    ax.plot(
        centers[k][0], 
        centers[k][1], 
        "o", 
        markerfacecolor='#CCC', 
        markeredgecolor=col, 
        markersize=9
    )

ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_linestyle('dotted')
ax.spines['left'].set_linestyle('dotted')

ax.xaxis.grid(True, linestyle='--', alpha=0.5)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

k_means = KMeans(n_clusters=3, random_state=808, n_init='auto').fit(X)

print("Centres des clusters détectés par KMeans :")
print(k_means.cluster_centers_)

k_means_labels = k_means.predict(X)

sil_score = silhouette_score(X, k_means_labels)
print(f"Silhouette score: {sil_score}")
