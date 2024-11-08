from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression  
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split

data_bunch = load_breast_cancer()
print(data_bunch.keys())

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=808)

clf = LogisticRegression(random_state=808, max_iter=10000).fit(X_train, y_train)

prediction_8 = clf.predict([X_test[8, :]])
prediction_13 = clf.predict([X_test[13, :]])

probability_8 = clf.predict_proba([X_test[8, :]])
probability_13 = clf.predict_proba([X_test[13, :]])

print("Probabilité pour l'observation 8 : ", probability_8)
print("Probabilité pour l'observation 13 : ", probability_13)

y_hat_proba = clf.predict_proba(X_test)[:, 1]
sns.histplot(y_hat_proba, kde=True)
plt.title("Distribution des probabilités prédites pour la classe positive")
plt.xlabel("Probabilité prédite")
plt.ylabel("Fréquence")
plt.show()

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :\n", conf_matrix)

y_pred_03 = [0 if value < 0.3 else 1 for value in y_hat_proba]
y_pred_07 = [0 if value < 0.7 else 1 for value in y_hat_proba]

conf_matrix_03 = confusion_matrix(y_test, y_pred_03)
conf_matrix_07 = confusion_matrix(y_test, y_pred_07)
print("Matrice de confusion avec seuil 0.3 :\n", conf_matrix_03)
print("Matrice de confusion avec seuil 0.7 :\n", conf_matrix_07)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Précision : ", precision)
print("Rappel : ", recall)

f1 = f1_score(y_test, y_pred)
print("Score F1 : ", f1)

fpr, tpr, thresholds = roc_curve(y_test, y_hat_proba)
plt.plot(fpr, tpr, label="Courbe ROC")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC")
plt.legend()
plt.show()
