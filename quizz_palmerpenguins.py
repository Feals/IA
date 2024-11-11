import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, mean_absolute_percentage_error, root_mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, f1_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression  
from sklearn.cluster import KMeans




filename = "https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/palmer_penguins_openclassrooms.csv"
data = pd.read_csv(filename)
print(data.head())



# question 1
X1 = data['bill_length_mm'].values.reshape(-1, 1)
y = data['body_mass_g']
reg1 = LinearRegression()
reg1.fit(X1, y)
score_bill_length = reg1.score(X1, y)

X2 = data['bill_depth_mm'].values.reshape(-1, 1)
reg2 = LinearRegression()
reg2.fit(X2, y)
score_bill_depth = reg2.score(X2, y)

X3 = data['flipper_length_mm'].values.reshape(-1, 1)
reg3 = LinearRegression()
reg3.fit(X3, y)
score_flipper_length = reg3.score(X3, y)

print("R² pour bill_length_mm:", score_bill_length)
print("R² pour bill_depth_mm:", score_bill_depth)
print("R² pour flipper_length_mm:", score_flipper_length)


# question 2
scaler = MinMaxScaler()
X = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm']])
y_scaled = scaler.fit_transform(data[['body_mass_g']])
df_scaled = pd.DataFrame(X, columns=['bill_length_mm','bill_depth_mm','flipper_length_mm'])
df_scaled['body_mass_g'] = y_scaled

X = df_scaled[['bill_length_mm','bill_depth_mm','flipper_length_mm']]
y = df_scaled['body_mass_g']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_hat_test = reg.predict(X_test)

print(f"Coefficients (normalisé) : {reg.coef_}")
print(f"RMSE (normalisé) : {root_mean_squared_error(y_test, y_hat_test)}")
print(f"MAPE (normalisé) : {mean_absolute_percentage_error(y_test, y_hat_test)}")


# question 3
for espece in ['Adelie', 'Gentoo', 'Chinstrap']:
    df = data[data.species == espece].copy()
    y = df['body_mass_g']
    X = scaler.fit_transform(df[['bill_length_mm','bill_depth_mm','flipper_length_mm']])
    reg.fit(X, y)
    print("--\n",espece, reg.score(X, y))

    y_pred = reg.predict(X)
    print(f"RMSE: {root_mean_squared_error(y, y_pred)}")
    print(f"MAPE: {mean_absolute_percentage_error(y, y_pred)}")

# question 4
y = data['body_mass_g']
X = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm']])

score = []
test_size = 0.2

for random_state in np.arange(200):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    reg.fit(X_train, y_train)
    print("--\n",test_size, reg.score(X_test, y_test))
    score.append(reg.score(X_test, y_test))


fig = plt.figure(figsize=(6, 6))
sns.boxplot(score)
plt.title(f"test_size {test_size}")
plt.show()

# question 5
filename = "https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/palmer_penguins_openclassrooms.csv"
data = pd.read_csv(filename)

data.loc[data.sex == 'male', 'sex'] = 0
data.loc[data.sex == 'female', 'sex'] = 1
data.dropna(inplace=True)
data['sex'] = data.sex.astype('int')
data['sex']

data['sex'].value_counts()

y = data['sex'].values
X = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm', 'body_mass_g']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

clf = LogisticRegression(random_state = 42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("Matrice de confusion :\n",conf_matrix)
print("tn :\n",tn)
print("fp :\n",fp)
print("fn :\n",fn)
print("tp :\n",tp)

# question 6
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))

# question 7
y_proba = clf.predict_proba(X_test)[:,1]
y_pred_03 = [ 0 if value < 0.3 else 1 for value in y_proba ]
y_pred_07 = [ 0 if value < 0.7 else 1 for value in y_proba ]

conf_matrix_03 = confusion_matrix(y_test, y_pred_03)
conf_matrix_07 = confusion_matrix(y_test, y_pred_07)
print("Matrice de confusion avec seuil 0.3 :\n", conf_matrix_03)
print("Matrice de confusion avec seuil 0.7 :\n", conf_matrix_07)

# question 8

data.loc[data.species == 'Adelie', 'species'] = 3
data.loc[data.species == 'Gentoo', 'species'] = 2
data.loc[data.species == 'Chinstrap', 'species'] = 1
data['species'] = data.species.astype('int')

y = data['species'].values
X1 = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm', 'body_mass_g']])
X1_train, X1_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X2 = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm']])
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.20, random_state=42)

clf1 = LogisticRegression(random_state = 42)
clf1.fit(X1_train, y_train)
clf2 = LogisticRegression(random_state = 42)
clf2.fit(X2_train, y_train)

y1_pred = clf1.predict(X1_test)
conf_matrix_y1 = confusion_matrix(y_test, y1_pred)
print("Matrice de confusion y1 :\n", conf_matrix_y1)
y2_pred = clf2.predict(X2_test)
conf_matrix_y2 = confusion_matrix(y_test, y2_pred)
print("Matrice de confusion y2 :\n", conf_matrix_y2)

# question 9

X = data[['bill_length_mm','bill_depth_mm','flipper_length_mm', 'body_mass_g','sex']]


km = KMeans( n_clusters=3, random_state = 808, n_init = 10)
km.fit(X)
y_pred = km.labels_
data['labels'] = km.labels_

result = data[['species', 'labels', 'island']].groupby(by = ['species', 'labels']).count().reset_index().rename(columns = {'island': 'count_'})

print(result)

scores = []
for n in range(2, 11, 1):
    km = KMeans( n_clusters=n, random_state = 808, n_init = 10)
    km.fit(X)
    labels_ = km.predict(X)
    scores.append(silhouette_score(X,labels_ ))
    print("silhouette_score: ", scores)

plt.plot(range(2, 11, 1), scores)
plt.title('Scores de Silhouette pour différents nombres de clusters')
plt.xlabel('Nombre de clusters')
plt.ylabel('Silhouette score')
plt.show()