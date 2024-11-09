import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

filename = "https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/palmer_penguins_openclassrooms.csv"
data = pd.read_csv(filename)
print(data.head())




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

