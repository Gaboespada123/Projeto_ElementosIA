import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Carregar o dataset já limpo
df = pd.read_csv("resultados/dataset_limpo.csv")

print("\nTamanho do dataset limpo:", df.shape)

# 2. Remover colunas que não devem entrar no modelo
# IDs servem para identificar registos, mas não ajudam na previsão
# record_created_at é uma data e não vai ser usada nesta versão simples
colunas_a_remover = [
    "program_id",
    "patient_id",
    "diet_id",
    "nutritionist_id",
    "record_created_at"
]

df = df.drop(columns=colunas_a_remover, errors="ignore")

print("\nColunas após remover identificadores e data:")
print(df.columns.tolist())

# 3. Separar a variável-alvo ANTES de qualquer transformação
# IMPORTANTE: não escalar o target para que as previsões sejam interpretáveis em kg
TARGET_COL = "weight_change_kg_6m"
y_target   = df[[TARGET_COL]].copy() if TARGET_COL in df.columns else None
df_features = df.drop(columns=[c for c in [TARGET_COL] if c in df.columns])

# 3.5 Limpar nomes das colunas e valores categoricos para evitar OHE duplicado
df_features.columns = df_features.columns.str.strip().str.lower().str.replace(" ", "_")

# 4. Separar colunas numericas e categoricas (apenas features)
colunas_numericas   = df_features.select_dtypes(include=["int64", "float64"]).columns.tolist()
colunas_categoricas = df_features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

for col in colunas_categoricas:
    df_features[col] = df_features[col].astype(str).str.strip().str.lower().str.replace(" ", "_")

print("\nColunas numéricas (features):")
print(colunas_numericas)

print("\nColunas categóricas:")
print(colunas_categoricas)

# 5. Aplicar StandardScaler às colunas numéricas (apenas features)
scaler = StandardScaler()
df_features_scaled = df_features.copy()
df_features_scaled[colunas_numericas] = scaler.fit_transform(df_features[colunas_numericas])

# 6. Aplicar One-Hot Encoding às colunas categóricas
df_final = pd.get_dummies(df_features_scaled, columns=colunas_categoricas, drop_first=True)

# 7. Reintroduzir o target (não escalado) no final
if y_target is not None:
    df_final = pd.concat([df_final, y_target], axis=1)

# 8. Mostrar dimensão final
print("\nTamanho do dataset final transformado:", df_final.shape)
print(f"Target '{TARGET_COL}' em escala original (kg), não escalado.")

# 9. Guardar o ficheiro final
df_final.to_csv("resultados/dados_limpos_final.csv", index=False)

print("\nFicheiro guardado em resultados/dados_limpos_final.csv")