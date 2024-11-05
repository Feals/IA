import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_csv("age_vs_poids_vs_taille_vs_sexe.csv")


X = df[['sexe', 'age', 'taille']]

y = df['poids']


reg = LinearRegression()


reg.fit(X, y)


print("Score du modèle:", reg.score(X, y))


print("Coefficients:", reg.coef_)


new_data = pd.DataFrame([[1, 150, 153]], columns=['sexe', 'age', 'taille'])
poids_pred = reg.predict(new_data)
print("Poids prédit:", poids_pred)
