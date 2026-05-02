"""
=============================================================
  SPRINT - TRANSFORMACIÓN DE DATOS PARA MACHINE LEARNING
=============================================================
  
  Input      : dataset_limpo.md  (tabla Markdown con pipes)
  Output     : dados_limpos_final.csv
  Librerías  : pandas, scikit-learn
=============================================================
"""
import os 
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# 1. CARGA DEL ARCHIVO CSV 
# ─────────────────────────────────────────────────────────────

import os
import pandas as pd

# 1. Obtenemos la ruta de la carpeta donde está guardado este script (.py)
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# 2. Unimos esa ruta dinámicamente con el nombre del archivo
ruta_exacta = os.path.join(directorio_actual, "dataset_limpo.csv")

# 3. Cargamos el CSV
df_raw = pd.read_csv(ruta_exacta)

print(f" Dataset cargado: {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas") 


# ─────────────────────────────────────────────────────────────
# 2. SEPARAR E IDENTIFICAR COLUMNAS POR TIPO
# ─────────────────────────────────────────────────────────────

# Columnas de IDs: no aportan información predictiva → se omiten
ID_COLS = ["program_id", "patient_id", "nutritionist_id", "diet_id"]

# Columnas de fecha: requieren ingeniería de features adicional → se omiten por ahora
DATE_COLS = ["record_created_at"]

COLS_TO_DROP = ID_COLS + DATE_COLS

# DataFrame de trabajo: sin IDs ni fechas
df = df_raw.drop(columns=[c for c in COLS_TO_DROP if c in df_raw.columns])

# Detección automática de columnas numéricas y categóricas
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
categ_cols   = df.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\n NUMERIC COLUMNS  ({len(numeric_cols)}): {numeric_cols}")
print(f" CATEGORICAL COLUMNS ({len(categ_cols)}): {categ_cols}")

# ─────────────────────────────────────────────────────────────
# 3. STANDARDSCALER → COLUMNAS NUMÉRICAS
# ─────────────────────────────────────────────────────────────
# StandardScaler transforma cada variable para que tenga
# media = 0 y desviación estándar = 1.
# Esto es imprescindible para modelos sensibles a la escala
# (regresión lineal, SVM, KNN, redes neuronales...).

scaler = StandardScaler()

df_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(df[numeric_cols]),
    columns=numeric_cols,
    index=df.index,
)

print(f"\n StandardScaler applied to {len(numeric_cols)} numeric columns.")

# ─────────────────────────────────────────────────────────────
# 4. ONE-HOT ENCODING → COLUMNAS CATEGÓRICAS
# ─────────────────────────────────────────────────────────────
# pd.get_dummies convierte cada categoría en una columna binaria (0/1).
#
# ▸ drop_first=True  → elimina una categoría por variable para evitar
#   multicolinealidad perfecta (dummy variable trap).
#   Importante para modelos lineales; puede omitirse para árboles.
#
# ▸ dtype=float       → mantiene consistencia de tipo con las columnas
#   escaladas, evitando problemas en frameworks como scikit-learn.
#
# Nota: normalizamos a minúsculas antes de codificar para unificar
# variantes como "Strict" y "strict" que son la misma categoría.

df_categ = df[categ_cols].copy()
df_categ = df_categ.apply(lambda col: col.str.strip().str.lower())

df_categ_encoded = pd.get_dummies(
    df_categ,
    columns=categ_cols,
    drop_first=True,
    dtype=float,
)

print(f" One-Hot Encoding applied to {len(categ_cols)} categorical columns.")
print(f"   Columns generated after OHE: {df_categ_encoded.shape[1]}")

# ─────────────────────────────────────────────────────────────
# 5. UNIÓN FINAL Y EXPORTACIÓN
# ─────────────────────────────────────────────────────────────
# Concatenamos los bloques horizontal mente (axis=1):
#   [ columnas numéricas escaladas | columnas OHE ]

df_final = pd.concat([df_numeric_scaled, df_categ_encoded], axis=1)

print(f"\n Final dataset: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
print(f"   ¿Valores nulos? {df_final.isnull().sum().sum()} (debería ser 0)")

# Exportar sin el índice de pandas
OUTPUT_PATH = "dados_limpos_final.csv"
df_final.to_csv(OUTPUT_PATH, index=False)

print(f"\n Archivo exportado correctamente → '{OUTPUT_PATH}'")
print("   ¡Dataset listo para entrenar modelos de Machine Learning! ")