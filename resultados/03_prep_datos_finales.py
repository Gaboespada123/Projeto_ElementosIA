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
# 2. SEPARAR E IDENTIFICAR COLUNAS POR TIPO
# ─────────────────────────────────────────────────────────────

# Coluna alvo: não deve ser escalada (mantém interpretação em kg)
TARGET_COL = "weight_change_kg_6m"

# Colunas de IDs: não aportam informação preditiva → removidas
ID_COLS = ["program_id", "patient_id", "nutritionist_id", "diet_id"]

# Colunas de data
DATE_COLS = ["record_created_at"]

COLS_TO_DROP = ID_COLS + DATE_COLS

# DataFrame de trabalho: sem IDs nem datas
df = df_raw.drop(columns=[c for c in COLS_TO_DROP if c in df_raw.columns])

# Separar target das features ANTES de qualquer transformação
y_target = df[[TARGET_COL]].copy() if TARGET_COL in df.columns else None
df_features = df.drop(columns=[c for c in [TARGET_COL] if c in df.columns])

# Deteção automática de colunas numéricas e categóricas (apenas features)
numeric_cols = df_features.select_dtypes(include=["number"]).columns.tolist()
categ_cols   = df_features.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\n NUMERIC FEATURE COLUMNS  ({len(numeric_cols)}): {numeric_cols}")
print(f" CATEGORICAL COLUMNS ({len(categ_cols)}): {categ_cols}")
print(f" TARGET (não escalado)     : {TARGET_COL}")

# ─────────────────────────────────────────────────────────────
# 3. STANDARDSCALER → COLUNAS NUMÉRICAS (apenas features, não o target)
# ─────────────────────────────────────────────────────────────
# StandardScaler transforma cada variável para ter média=0 e desvio padrão=1.
# IMPORTANTE: a variável-alvo weight_change_kg_6m NÃO é escalada,
# para que as previsões do modelo sejam interpretáveis em kg.

scaler = StandardScaler()

df_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(df_features[numeric_cols]),
    columns=numeric_cols,
    index=df_features.index,
)

print(f"\n StandardScaler aplicado a {len(numeric_cols)} colunas numéricas (features apenas).")

# ─────────────────────────────────────────────────────────────
# 4. ONE-HOT ENCODING → COLUNAS CATEGÓRICAS
# ─────────────────────────────────────────────────────────────
# pd.get_dummies converte cada categoria numa coluna binária (0/1).
# drop_first=True elimina uma categoria por variável para evitar
# multicolinearidade perfeita (dummy variable trap).
# Normalizamos para minúsculas antes de codificar para unificar
# variantes como "Strict" e "strict" que são a mesma categoria.

df_categ = df_features[categ_cols].copy()
df_categ = df_categ.apply(lambda col: col.str.strip().str.lower())

df_categ_encoded = pd.get_dummies(
    df_categ,
    columns=categ_cols,
    drop_first=True,
    dtype=float,
)

print(f" One-Hot Encoding aplicado a {len(categ_cols)} colunas categóricas.")
print(f"   Colunas geradas após OHE: {df_categ_encoded.shape[1]}")

# ─────────────────────────────────────────────────────────────
# 5. UNIÃO FINAL E EXPORTAÇÃO
# ─────────────────────────────────────────────────────────────
# Concatenamos os blocos horizontalmente (axis=1):
#   [ features numéricas escaladas | colunas OHE | target NÃO escalado ]
# O target fica NO FINAL para facilitar a separação X / y nos scripts de modelos.

if y_target is not None:
    df_final = pd.concat([df_numeric_scaled, df_categ_encoded, y_target], axis=1)
else:
    df_final = pd.concat([df_numeric_scaled, df_categ_encoded], axis=1)

print(f"\n Dataset final: {df_final.shape[0]} linhas × {df_final.shape[1]} colunas")
print(f"   Valores nulos? {df_final.isnull().sum().sum()} (deve ser 0)")
print(f"   O target '{TARGET_COL}' está em escala original (kg), não escalado.")

# Exportar sem o índice do pandas
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultados", "dados_limpos_final.csv")
df_final.to_csv(OUTPUT_PATH, index=False)

print(f"\n Ficheiro exportado → 'resultados/dados_limpos_final.csv'")
print("   Dataset pronto para treinar modelos de Machine Learning!")