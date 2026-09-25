
# 4. Carga y exploración de los datos

import numpy as np
import pandas as pd
# Obtiene el dataset de abalone desde un repositorio en GitHub
url = "https://raw.githubusercontent.com/juanluisMerida23/Machine-Learning-Abalone/refs/heads/main/Dataset/abalone.data"
#importamos diciendo que no tiene cabecera y le damos nombre a las columnas
 
df = pd.read_csv(
    url,
    header=None,
    names=[
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
        "Rings"
    ]
)

# Muestra las primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Indica el número de filas y columnas
print("\nNúmero de filas y columnas:")
print(df.shape)

# Muestra los nombres de las columnas
print("\nNombres de las columnas:")
print(df.columns.tolist())

# Comprueba los tipos de datos
print("\nTipos de datos:")
print(df.dtypes)

# Comprueba si existen valores nulos
print("\nValores nulos:")
print(df.isnull().sum())

# Obtén estadísticas básicas de las variables
print("\nEstadísticas básicas:")
print(df.describe())

print("""
📊 RESUMEN DEL CONJUNTO DE DATOS

El conjunto de datos contiene 9 variables:

• Sex → Variable categórica que indica el sexo del abalón:
    - M = Macho
    - F = Hembra
    - I = Juvenil (Infant)

• Length, Diameter, Height, Whole_weight,
  Shucked_weight, Viscera_weight y Shell_weight
  → 7 variables numéricas continuas (float64)
    relacionadas con medidas y pesos del abalón.

• Rings → Variable numérica entera (int64)
    que representa el número de anillos de crecimiento.

🎯 La variable 'Rings' permite estimar la edad del ejemplar:
   Edad ≈ Rings + 1.5 años
""")

# 5. Análisis de los datos

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# GRÁFICO 1: RELACIÓN ENTRE UNA VARIABLE FÍSICA Y LA EDAD
# ==========================================

print("""
📈 GRÁFICO 1: RELACIÓN ENTRE EL PESO DE LA CONCHA Y LA EDAD

Este gráfico de dispersión muestra la relación entre:

• Shell_weight → Peso de la concha.
• Rings → Número de anillos (indicador de la edad).

Cada punto representa un abalón del conjunto de datos.

Objetivo:
Analizar si existe una relación entre el peso de la concha y la edad.
Si los puntos muestran una tendencia ascendente, significará que los
abalones con conchas más pesadas suelen tener más anillos y, por tanto,
mayor edad.
""")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Shell_weight", y="Rings", alpha=0.5)

plt.title("Relación entre el peso de la concha y los anillos")
plt.xlabel("Peso de la concha")
plt.ylabel("Número de anillos")
plt.grid(True)

plt.show()

# ==========================================
# MATRIZ DE CORRELACIONES
# ==========================================

print("""
📊 GRÁFICO 2: MATRIZ DE CORRELACIONES

La matriz de correlaciones permite medir el grado de relación lineal
entre todas las variables numéricas del conjunto de datos.

Interpretación de los valores:

• Correlación cercana a 1  → Relación positiva fuerte.
• Correlación cercana a 0  → Relación débil o inexistente.
• Correlación cercana a -1 → Relación negativa fuerte.

En el mapa de calor:

🔴 Colores cálidos → Relación positiva más fuerte.
🔵 Colores fríos   → Relación negativa más fuerte.

El objetivo es identificar qué variables están más relacionadas con
'Rings', que será la variable a predecir mediante regresión lineal.
""")

# Excluimos la variable categórica Sex
correlaciones = df.drop("Sex", axis=1).corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlaciones,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Matriz de correlaciones")
plt.show()

# ==========================================
# VARIABLES MÁS RELACIONADAS CON RINGS
# ==========================================

print("""
📋 ANÁLISIS DE LAS VARIABLES MÁS RELACIONADAS CON LA EDAD

A continuación se calcula la correlación de cada variable con 'Rings'.

Cuanto más próximo esté el valor a 1 o -1, más fuerte será la relación
con la edad del abalón.

Estas variables serán las más útiles para construir posteriormente
un modelo de regresión lineal capaz de predecir la edad.
""")

corr_rings = correlaciones["Rings"].sort_values(ascending=False)

print("\nCorrelación de cada variable con Rings:")
print(corr_rings)

print("\nRanking de variables más relacionadas con Rings:")

ranking = (
    correlaciones["Rings"]
    .drop("Rings")
    .abs()
    .sort_values(ascending=False)
)

for i, (variable, valor) in enumerate(ranking.items(), start=1):
    print(f"{i}. {variable}: {valor:.3f}")
    
# 6.  Preparación de los datos

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("""
==================================================
🛠 PREPARACIÓN DE LOS DATOS
==================================================

Objetivo:
Preparar el conjunto de datos para entrenar un modelo
de regresión lineal capaz de predecir la edad del abalón.

Pasos realizados:

1️⃣ Definir las variables predictoras (X).
   Se utilizarán todas las características físicas
   disponibles en el conjunto de datos.

2️⃣ Definir la variable objetivo (y).
   Se utilizará la variable 'Rings', ya que representa
   el número de anillos de crecimiento y permite estimar
   la edad del abalón.

3️⃣ Transformar la variable categórica 'Sex'.
   Los algoritmos de Machine Learning trabajan con datos
   numéricos, por lo que los valores categóricos deben
   convertirse previamente a números.

4️⃣ Dividir el conjunto de datos.
   • 80 % para entrenamiento.
   • 20 % para prueba.

==================================================
""")

# ==========================================
# TRANSFORMACIÓN DE LA VARIABLE CATEGÓRICA
# ==========================================

le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])

print("""
==================================================
🔄 TRANSFORMACIÓN DE LA VARIABLE CATEGÓRICA
==================================================

La variable 'Sex' contiene valores de texto:

• M = Macho
• F = Hembra
• I = Juvenil

Para poder utilizar esta información en el modelo,
se ha transformado a valores numéricos mediante
LabelEncoder.
==================================================
""")

codificacion = dict(zip(le.classes_, le.transform(le.classes_)))

print("Codificación aplicada:")
for categoria, codigo in codificacion.items():
    print(f"• {categoria} = {codigo}")

# ==========================================
# DEFINICIÓN DE X E y
# ==========================================

X = df.drop("Rings", axis=1)
y = df["Rings"]

print("""
==================================================
📋 DEFINICIÓN DE VARIABLES
==================================================

Variable objetivo (y):
• Rings

Variables predictoras (X):
""")

for columna in X.columns:
    print(f"• {columna}")

# ==========================================
# DIVISIÓN DE LOS DATOS
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"""
==================================================
📊 DIVISIÓN DEL CONJUNTO DE DATOS
==================================================

Número total de registros: {len(df)}

Conjunto de entrenamiento (80 %):
• X_train: {X_train.shape}
• y_train: {y_train.shape}

Conjunto de prueba (20 %):
• X_test : {X_test.shape}
• y_test : {y_test.shape}

El conjunto de entrenamiento se utilizará para
aprender los patrones presentes en los datos.

El conjunto de prueba se utilizará posteriormente
para evaluar la capacidad predictiva del modelo.

==================================================
""")

print("""
==================================================
✅ CONCLUSIÓN
==================================================

Se ha seleccionado 'Rings' como variable objetivo
porque permite estimar la edad del abalón.

Como variables predictoras se han utilizado todas
las características físicas disponibles en el dataset.

La variable categórica 'Sex' ha sido transformada
a valores numéricos para que pueda ser procesada
por los algoritmos de Machine Learning.

Finalmente, los datos se han dividido en un 80 %
para entrenamiento y un 20 % para prueba, quedando
todo preparado para entrenar y evaluar un modelo
de regresión lineal.

==================================================
""")
    
# 7. Creación y entrenamiento del modelo

# 7. Creación y entrenamiento del modelo

from sklearn.linear_model import LinearRegression

print("""
==================================================
🤖 CREACIÓN Y ENTRENAMIENTO DEL MODELO
==================================================

Para este problema se ha seleccionado un modelo de
Regresión Lineal mediante Scikit-Learn.

La variable objetivo ('Rings') es numérica, por lo
que la regresión lineal resulta adecuada para
predecirla a partir de las características físicas
del abalón.

Pasos realizados:

1. Crear el modelo.
2. Entrenar el modelo con los datos de entrenamiento.
3. Realizar predicciones sobre los datos de prueba.

==================================================
""")

# Crear modelo
modelo = LinearRegression()

print("✅ Modelo creado correctamente.")

# Entrenar modelo
modelo.fit(X_train, y_train)

print("✅ Modelo entrenado correctamente.")

# Realizar predicciones
y_pred = modelo.predict(X_test)

print("""
✅ Predicciones realizadas correctamente.

Ejemplos de comparación entre valores reales y
valores predichos:
""")

y_test_reset = y_test.reset_index(drop=True)

print("\nPrimeras 5 comparaciones:")

for i, (real, pred) in enumerate(zip(y_test.head(5), y_pred[:5]), start=1):
    print(f"Registro {i}: Real = {real:.2f} | Predicción = {pred:.2f}")

print("""
==================================================
✅ PROCESO FINALIZADO
==================================================

El modelo ha sido creado y entrenado utilizando
los datos de entrenamiento.

Posteriormente se han realizado predicciones sobre
el conjunto de prueba.

El siguiente paso consistirá en evaluar la calidad
de dichas predicciones mediante distintas métricas.
==================================================
""")
# 8. Evaluación del modelo
# 8. Evaluación del modelo

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("""
==================================================
📊 EVALUACIÓN DEL MODELO
==================================================

Objetivo:

Evaluar la calidad de las predicciones realizadas por
el modelo de Regresión Lineal utilizando el conjunto
de datos de prueba.

Métricas utilizadas:

• MAE  (Error Absoluto Medio)
• MSE  (Error Cuadrático Medio)
• RMSE (Raíz del Error Cuadrático Medio)
• R²   (Coeficiente de Determinación)

==================================================
""")

# ==========================================
# CÁLCULO DE MÉTRICAS
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ==========================================
# RESULTADOS
# ==========================================

print("RESULTADOS OBTENIDOS:\n")

print(f"MAE  : {mae:.3f}")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# ==========================================
# INTERPRETACIÓN
# ==========================================

print("""
==================================================
📝 INTERPRETACIÓN DE LOS RESULTADOS
==================================================

MAE:
Indica el error medio que comete el modelo en sus
predicciones. Cuanto menor sea este valor, mejor.

MSE:
Penaliza más los errores grandes al elevarlos al
cuadrado. Cuanto menor sea, mejor será el modelo.

RMSE:
Representa el error medio en las mismas unidades
que la variable objetivo ('Rings'). Cuanto menor
sea, mejor.

R²:
Mide la capacidad explicativa del modelo.

• R² = 1  --> Predicción perfecta.
• R² = 0  --> No mejora respecto a usar la media.
• R² < 0  --> Modelo muy deficiente.

En general, un valor de R² más cercano a 1 y errores
(MAE, MSE y RMSE) más bajos indican un mejor ajuste
del modelo a los datos.

==================================================
""")

if r2 >= 0.70:
    print("✅ El modelo presenta un buen rendimiento.")
elif r2 >= 0.50:
    print("✅ El modelo presenta un rendimiento aceptable.")
else:
    print("⚠️ El modelo presenta un rendimiento limitado y podría mejorarse.")