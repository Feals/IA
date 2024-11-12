import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("paris_trees.csv")
print(df.head())
print(df.describe())
df = df[df.libelle_francais == 'Platane'].copy()

df['hauteur_m'].replace(0, df['hauteur_m'].median(), inplace=True)
df['circonference_cm'].replace(0, df['circonference_cm'].median(), inplace=True)
categorie_order = ["Jeune (arbre)", "Jeune (arbre)Adulte", "Adulte", "Mature"]
df['stade_de_developpement'] = pd.Categorical(df['stade_de_developpement'], categories=categorie_order, ordered=True)

sns.boxplot(data= df, y="circonference_cm", x="stade_de_developpement", order=categorie_order)
plt.show()
sns.boxplot(data=df, y="hauteur_m", x="stade_de_developpement", order=categorie_order)
plt.show()

cond_mature = (df.stade_de_developpement.isna()) & (df.hauteur_m > 20) & (df.circonference_cm > 200)
df[cond_mature].shape
df.loc[cond_mature, 'stade_de_developpement'] = "Mature"

cond_jeune = (df.stade_de_developpement.isna()) & (df.hauteur_m < 8) & (df.circonference_cm < 50)
df[cond_jeune].shape
df.loc[cond_jeune, 'stade_de_developpement'] = "Jeune (arbre)"

df = df[(df.circonference_cm < 1000) & (df.hauteur_m < 100)].copy()
df['z_circonference'] = stats.zscore(df.circonference_cm)
df['z_hauteur'] = stats.zscore(df.hauteur_m)
df = df[(df['z_circonference'].abs() < 2) & (df['z_hauteur'].abs() < 2)]

iqr = np.quantile(df.hauteur_m, q=[0.25, 0.75])
limite_basse = iqr[0] - 1.5*(iqr[1] - iqr[0])
limite_haute = iqr[1] + 1.5*(iqr[1] - iqr[0])

scaler = StandardScaler()
df['hauteur_standard'] = scaler.fit_transform(df.hauteur_m.values.reshape(-1, 1))
df['circonference_standard'] = scaler.fit_transform(df.circonference_cm.values.reshape(-1, 1)) 

print(df.libelle_francais.value_counts())

df_marronnier = df[df.libelle_francais == 'Marronnier'].copy()
df_tilleul = df[df.libelle_francais == 'Tilleul'].copy()

df_marronnier['hauteur_m'].replace(0, df_marronnier['hauteur_m'].median(), inplace=True)
df_marronnier['circonference_cm'].replace(0, df_marronnier['circonference_cm'].median(), inplace=True)

df_tilleul['hauteur_m'].replace(0, df_tilleul['hauteur_m'].median(), inplace=True)
df_tilleul['circonference_cm'].replace(0, df_tilleul['circonference_cm'].median(), inplace=True)

df_marronnier['z_circonference'] = stats.zscore(df_marronnier.circonference_cm)
df_marronnier['z_hauteur'] = stats.zscore(df_marronnier.hauteur_m)
df_marronnier = df_marronnier[(df_marronnier['z_circonference'].abs() < 2) & (df_marronnier['z_hauteur'].abs() < 2)]

df_tilleul['z_circonference'] = stats.zscore(df_tilleul.circonference_cm)
df_tilleul['z_hauteur'] = stats.zscore(df_tilleul.hauteur_m)
df_tilleul = df_tilleul[(df_tilleul['z_circonference'].abs() < 3) & (df_tilleul['z_hauteur'].abs() < 3)]

df_marronnier['hauteur_standard'] = scaler.fit_transform(df_marronnier.hauteur_m.values.reshape(-1, 1))
df_marronnier['circonference_standard'] = scaler.fit_transform(df_marronnier.circonference_cm.values.reshape(-1, 1))

df_tilleul['hauteur_standard'] = scaler.fit_transform(df_tilleul.hauteur_m.values.reshape(-1, 1))
df_tilleul['circonference_standard'] = scaler.fit_transform(df_tilleul.circonference_cm.values.reshape(-1, 1))

sns.boxplot(data=df_marronnier, y="circonference_cm", x="stade_de_developpement", order=categorie_order)
plt.title("Circonférence des Marronniers par stade de développement")
plt.show()

sns.boxplot(data=df_tilleul, y="circonference_cm", x="stade_de_developpement", order=categorie_order)
plt.title("Circonférence des Tilleuls par stade de développement")
plt.show()