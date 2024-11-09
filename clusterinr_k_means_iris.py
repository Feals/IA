from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
X = iris.data
y = iris.target

model = KMeans(n_clusters=3, n_init="auto")
model.fit(X)

print("score", model.score(X))
print("silhouette_score: ", silhouette_score(X,model.labels_ ))

labels = model.labels_.copy()
labels[labels==0] = 5
labels[labels==1] = 10
labels[labels==2] = 15

labels[labels == 5] = 0
labels[labels == 10] = 2
labels[labels == 15] = 1


print("accuracy_score",accuracy_score(y,labels ))
print("confusion_matrix\n",confusion_matrix(y,labels ))
