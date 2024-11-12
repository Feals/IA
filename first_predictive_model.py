import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error



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

y_pred = reg.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mape = mean_absolute_percentage_error(y, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Mean Absolute Percentage Error:", mape)