import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("advertising.csv")
print(df.head())
print(df.describe())

sns.regplot(x="tv", y="ventes", data=df)
plt.title("Relation entre TV et Ventes")
plt.show()

sns.regplot(x="radio", y="ventes", data=df)
plt.title("Relation entre Radio et Ventes")
plt.show()

sns.regplot(x="journaux", y="ventes", data=df)
plt.title("Relation entre Journaux et Ventes")
plt.show()

print("Corrélations :")
print(df.corr())


reg = LinearRegression()

X = df[['tv', 'radio', 'journaux']]
y = df['ventes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg.fit(X_train, y_train)

y_pred_test = reg.predict(X_test)
print("y_pred", y_pred_test)
print(f"Coefficients : {reg.coef_}")
print(f"RMSE : {root_mean_squared_error(y_test, y_pred_test)}")
print(f"MAPE : {mean_absolute_percentage_error(y_test, y_pred_test)}")

df['tv2'] = df['tv'] ** 2
scaler = MinMaxScaler()
data_array = scaler.fit_transform(df[['tv', 'radio', 'journaux', 'ventes', 'tv2']])
df_scaled = pd.DataFrame(data_array, columns=['tv', 'radio', 'journaux', 'ventes', 'tv2'])

X = df_scaled[['tv', 'radio', 'journaux', 'tv2']]
y = df_scaled['ventes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg.fit(X_train, y_train)

y_hat_test = reg.predict(X_test)

print(f"Coefficients (normalisé) : {reg.coef_}")
print(f"RMSE (normalisé) : {root_mean_squared_error(y_test, y_hat_test)}")
print(f"MAPE (normalisé) : {mean_absolute_percentage_error(y_test, y_hat_test)}")


df['tv_radio'] = df['tv'] * df['radio']

regressions = {
    'simple: y ~ tv + radio + journaux'  :     ['tv','radio','journaux'],
    'quadratique: y ~ tv + radio + journaux + tv2': ['tv','radio','journaux', 'tv2'],
    'terme croisée: y ~ tv + radio + journaux + tv*radio':['tv','radio','journaux', 'tv_radio']
}

for title, variables in regressions.items():
    scaler = MinMaxScaler()
    data_array = scaler.fit_transform(df[variables])

    X_train, X_test, y_train, y_test = train_test_split(data_array, y, test_size=0.20, random_state=42)

    reg.fit(X_train, y_train)
    y_pred_test = reg.predict(X_test)

    print(f"\n-- Regression {title}")
    print(f"\tRMSE: {root_mean_squared_error(y_test, y_pred_test)}")
    print(f"\tMAPE: {mean_absolute_percentage_error(y_test, y_pred_test)}")

new_data = pd.DataFrame([[153, 26, 74]], columns=['tv', 'radio', 'journaux'])
new_data['tv_radio'] = new_data['tv'] * new_data['radio']
ventes_pred = reg.predict(new_data[['tv', 'radio', 'journaux', 'tv_radio']].to_numpy())

print("Ventes prédites:", ventes_pred)