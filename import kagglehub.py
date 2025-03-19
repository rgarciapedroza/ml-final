import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('Diabetes Missing Data (1).csv')
print(df.head())

"""CHECKING QUANTITY OF MISSING VALUES"""

missing_data = df.isnull().sum()
total_entries = len(df)
print(total_entries)
print(missing_data)
missing_entries = df.isnull().any(axis=1).sum()
print(total_entries-missing_entries)

X = df.drop(columns=['Class'])
y = df['Class']

#PLOT OF MISSING VALUES
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Mapa de valores faltantes en el dataset")
#plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def train_and_evaluate_no_predata_proccess_changing_n_estimators(n_estimators, X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {n_estimators} estimators: {accuracy:.4f}")
    return accuracy

no_predata_proccess_changing_n_estimators_list = [1, 5, 15, 50, 100, 200, 500, 1000]
no_predata_proccess_changing_n_estimators_accuracies = [train_and_evaluate_no_predata_proccess_changing_n_estimators(n, X_train, X_test, y_train, y_test) for n in no_predata_proccess_changing_n_estimators_list]

plt.figure(figsize=(10, 6))
plt.plot([f"{n} Est." for n in no_predata_proccess_changing_n_estimators_list], no_predata_proccess_changing_n_estimators_accuracies, marker='o', linestyle='-', color='b')
plt.title("Accuracy de los modelos con diferentes n_estimators", fontsize=14)
plt.xlabel("Modelos", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


def train_and_evaluate_no_predata_proccess_changing_max_depth(max_depth, X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(100, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with max_depth={max_depth}: {accuracy:.4f}")
    return accuracy


no_predata_proccess_changing_max_depth_list = [None, 1, 5, 10, 12, 15, 17]
no_predata_proccess_changing_max_depth_accuracy = [train_and_evaluate_no_predata_proccess_changing_max_depth( d, X_train, X_test, y_train, y_test) for d in no_predata_proccess_changing_max_depth_list]

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.plot([f"max_depth={d}" for d in no_predata_proccess_changing_max_depth_list], no_predata_proccess_changing_max_depth_accuracy, marker='o', linestyle='-', color='b')
plt.title("Accuracy de los modelos con diferentes max_depth", fontsize=14)
plt.xlabel("Modelos", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

def train_and_evaluate_no_predata_proccess_changing_min_samples_leaf(min_samples_leaf, X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=min_samples_leaf, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with max_depth=1, min_samples_leaf={min_samples_leaf}: {accuracy:.4f}")
    return accuracy

no_predata_proccess_changing_min_samples_leaf_list = [ 1, 3, 5, 10, 12, 15, 17]
no_predata_proccess_changing_min_samples_leaf_accuracy = [train_and_evaluate_no_predata_proccess_changing_min_samples_leaf( d, X_train, X_test, y_train, y_test) for d in no_predata_proccess_changing_min_samples_leaf_list]

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.plot([f"mean_samples_leaf={d}" for d in no_predata_proccess_changing_min_samples_leaf_list], no_predata_proccess_changing_min_samples_leaf_accuracy, marker='o', linestyle='-', color='b')
plt.title("Accuracy de los modelos con diferentes min_samples_leaf", fontsize=14)
plt.xlabel("Modelos", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

"""
EXPLICACION

Valores faltantes no tratados:

Muchas columnas (age, bmi, blood_pressure, glucose_levels) tienen una gran cantidad de datos faltantes (4555, 5348, 6234 y 5244 respectivamente).

Esto reduce la cantidad efectiva de datos útiles para el entrenamiento y la predicción del modelo. El modelo puede estar aprendiendo principalmente de las otras columnas (gender, smoking_status).

Columna condition:

La columna condition es tu target (lo que estás tratando de predecir). Si el modelo no tiene suficiente información útil en las features restantes debido a los valores faltantes, su capacidad para mejorar el rendimiento será limitada.

Modelo insensible a hiperparámetros:

Debido al alto nivel de valores faltantes, el modelo puede estar entrenándose en un subconjunto pequeño de datos, lo que limita el impacto de ajustar los parámetros como max_depth o min_samples_leaf.

Poca variabilidad en las características restantes:

Si los datos útiles (por ejemplo, gender y smoking_status, que ya fueron convertidos a valores binarios) no contienen suficiente información predictiva, los ajustes en los parámetros no afectarán significativamente el modelo.

////////////////////////////////////////////////////////////////////////////////////////////////////////"""

#REPLACING MISSING VALUES WITH THE MEAN

df_mean = df.copy()
df_mean.fillna(df.mean(), inplace=True)

print(df_mean.isnull().sum())

#OBTAIN TARGET AND VALUES
X_mean = df_mean.drop(columns=['condition'])
y_mean = df_mean['condition']

X_train_mean, X_test_mean, y_train_mean, y_test_mean = train_test_split(X_mean, y_mean, test_size=0.3, random_state=42)

def train_and_evaluate_filling_with_the_mean_changing_n_estimators(n_estimators, X_train_mean, X_test_mean, y_train_mean, y_test_mean):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train_mean, y_train_mean)
    y_pred = model.predict(X_test_mean)
    accuracy = accuracy_score(y_test_mean, y_pred)
    print(f"Accuracy with {n_estimators} estimators: {accuracy:.4f}")
    return accuracy

filling_with_the_mean_changing_n_estimators_list = [1, 5, 15, 50, 100, 200, 500, 1000]
filling_with_the_mean_changing_n_estimators_accuracy = [train_and_evaluate_filling_with_the_mean_changing_n_estimators(n, X_train_mean, X_test_mean, y_train_mean, y_test_mean) for n in filling_with_the_mean_changing_n_estimators_list]

plt.figure(figsize=(10, 6))
plt.plot([f"{n} Est." for n in filling_with_the_mean_changing_n_estimators_list], filling_with_the_mean_changing_n_estimators_accuracy, marker='o', linestyle='-', color='b')
plt.title("Accuracy de los modelos con diferentes n_estimators", fontsize=14)
plt.xlabel("Modelos", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


def train_and_evaluate_filling_with_the_mean_changing_max_depth(max_depth, X_train_mean, X_test_mean, y_train_mean, y_test_mean):
    model = RandomForestClassifier(n_estimators=200, max_depth=max_depth, random_state=42)
    model.fit(X_train_mean, y_train_mean)
    y_pred = model.predict(X_test_mean)
    accuracy = accuracy_score(y_test_mean, y_pred)
    print(f"Accuracy with max_depth={max_depth}: {accuracy:.4f}")
    return accuracy

filling_with_the_mean_changing_max_depth_list = [None, 1, 3, 5, 10, 12, 15, 17]
filling_with_the_mean_changing_max_depth_accuracy = [train_and_evaluate_filling_with_the_mean_changing_max_depth(n, X_train_mean, X_test_mean, y_train_mean, y_test_mean) for n in filling_with_the_mean_changing_max_depth_list]

plt.figure(figsize=(10, 6))
plt.plot([f"{n} Est." for n in filling_with_the_mean_changing_max_depth_list], filling_with_the_mean_changing_max_depth_accuracy, marker='o', linestyle='-', color='b')
plt.title("Accuracy de los modelos con diferentes max_depth", fontsize=14)
plt.xlabel("Modelos", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

def train_and_evaluate_filling_with_the_mean_min_samples_leaf(min_samples_leaf, X_train_mean, X_test_mean, y_train_mean, y_test_mean):
    model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=min_samples_leaf, random_state=42)
    model.fit(X_train_mean, y_train_mean)
    y_pred_mean = model.predict(X_test_mean)
    accuracy = accuracy_score(y_test_mean, y_pred_mean)
    print(f"Accuracy with max_depth=5, min_samples_leaf={min_samples_leaf}: {accuracy:.4f}")
    return accuracy

filling_with_the_mean_min_samples_leaf_list = [50, 100, 500, 10, 12, 15, 17]
filling_with_the_mean_min_samples_leaf_accuracy = [train_and_evaluate_filling_with_the_mean_min_samples_leaf( d, X_train_mean, X_test_mean, y_train_mean, y_test_mean) for d in filling_with_the_mean_min_samples_leaf_list]

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.plot([f"mean_samples_leaf={d}" for d in filling_with_the_mean_min_samples_leaf_list], filling_with_the_mean_min_samples_leaf_accuracy, marker='o', linestyle='-', color='b')
plt.title("Accuracy de los modelos con diferentes min_samples_leaf", fontsize=14)
plt.xlabel("Modelos", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()