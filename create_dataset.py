from sklearn.datasets import make_classification
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles, make_moons
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=1000,  
    n_features=3,    
    n_informative=2, 
    n_redundant=0,   
    n_classes=2,     
    random_state=42 
)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X[:, 0], X[:, 1], c=y)
plt.grid()
plt.show()

df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2', 'Feature 3'])
df['Label'] = y
df.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(random_state=42).fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision de la régression logistique sur le dataset make_classification : {accuracy:.2f}")

X_circle, y_circle = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X_circle[:, 0], X_circle[:, 1], c=y_circle)
plt.grid()
plt.show()
X_moon, y_moon = make_moons(n_samples=1000, noise=0.1, random_state=42)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon)
plt.grid()
plt.show()

def train_and_evaluate_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X_circle, y_circle, random_state=42)
clf = LogisticRegression(random_state = 42).fit(X_train,y_train)

print(f"Score de la classification R^2 sur les cercles: ", clf.score(X_test, y_test))
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
ax.set_title('original')

y_pred = clf.predict(X_test)
ax = fig.add_subplot(1, 2, 2)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
ax.set_title('classification')

accuracy_circles = train_and_evaluate_logistic_regression(X_circle, y_circle)
print(f"Précision de la régression logistique sur le dataset make_circles : {accuracy_circles:.2f}")

X_train, X_test, y_train, y_test = train_test_split(X_moon, y_moon, random_state=0)

clf = LogisticRegression(random_state = 0).fit(X_train,y_train)

print(f"Score de la classification R^2 sur les moons: ", clf.score(X_test, y_test))

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
ax.set_title('original')

y_pred = clf.predict(X_test)
ax = fig.add_subplot(1, 2, 2)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
ax.set_title('classification')

accuracy_moons = train_and_evaluate_logistic_regression(X_moon, y_moon)
print(f"Précision de la régression logistique sur le dataset make_moons : {accuracy_moons:.2f}")