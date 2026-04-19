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

# 3. Separar colunas numéricas e categóricas
colunas_numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
colunas_categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nColunas numéricas:")
print(colunas_numericas)

print("\nColunas categóricas:")
print(colunas_categoricas)

# 4. Aplicar StandardScaler às colunas numéricas
scaler = StandardScaler()
df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

# 5. Aplicar One-Hot Encoding às colunas categóricas
df_final = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)

# 6. Mostrar dimensão final
print("\nTamanho do dataset final transformado:", df_final.shape)

# 7. Guardar o ficheiro final
df_final.to_csv("resultados/dados_limpos_final.csv", index=False)

print("\nFicheiro guardado em resultados/dados_limpos_final.csv")