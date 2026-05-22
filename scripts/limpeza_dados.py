import pandas as pd

# 1. Carregar os ficheiros CSV
patients = pd.read_csv("data/patients.csv")
diets = pd.read_csv("data/diets.csv")
nutritionists = pd.read_csv("data/nutritionists.csv")
outcomes = pd.read_csv("data/outcomes.csv")

# 2. Unir os datasets
# A tabela outcomes é usada como base porque liga paciente, dieta e nutricionista
df = outcomes.merge(patients, on="patient_id", how="left")
df = df.merge(diets, on="diet_id", how="left")
df = df.merge(nutritionists, on="nutritionist_id", how="left")

print("\nTamanho inicial do dataset unido:", df.shape)

# 3. Remover colunas redundantes
# bmi_redundant parece duplicar baseline_bmi
# experience_years parece duplicar years_experience
colunas_redundantes = ["bmi_redundant", "experience_years"]
df = df.drop(columns=colunas_redundantes, errors="ignore")

# 4. Preencher nulos em colunas categóricas com a moda
# Escolha simples e fácil de justificar
if "approach" in df.columns and df["approach"].isnull().sum() > 0:
    df["approach"] = df["approach"].fillna(df["approach"].mode()[0])

# 5. Preencher nulos em colunas numéricas com a média
# Só para variáveis de entrada normais
colunas_media = [
    "fiber_target_g",
    "sodium_limit_mg",
    "years_experience",
    "sleep_hours",
    "age",
    "motivation_score",
    "motivation_score_program"
]

for col in colunas_media:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# 6. Tratar a variável de resultado
# Não imputamos a variável-alvo para não inventar resultados
# Removemos linhas onde falta weight_change_kg_6m
if "weight_change_kg_6m" in df.columns:
    df = df.dropna(subset=["weight_change_kg_6m"])

# 7. Remover nulos em métricas derivadas relacionadas com o resultado
# Estas métricas fazem parte do desfecho do programa e não convém inventar valores
colunas_resultado_relacionadas = ["mean_adherence_pct", "adherence_ratio"]
colunas_existentes = [col for col in colunas_resultado_relacionadas if col in df.columns]
if colunas_existentes:
    df = df.dropna(subset=colunas_existentes)

# 8. Remover duplicados completos
df = df.drop_duplicates()

# 9. Remover outliers impossíveis com regras simples
# Regras fáceis de explicar ao professor
if "age" in df.columns:
    df = df[(df["age"] >= 10) & (df["age"] <= 100)]

if "height_cm" in df.columns:
    df = df[(df["height_cm"] >= 120) & (df["height_cm"] <= 230)]

if "baseline_weight_kg" in df.columns:
    df = df[(df["baseline_weight_kg"] >= 30) & (df["baseline_weight_kg"] <= 300)]

if "baseline_bmi" in df.columns:
    df = df[(df["baseline_bmi"] >= 10) & (df["baseline_bmi"] <= 80)]

if "sleep_hours" in df.columns:
    df = df[(df["sleep_hours"] >= 0) & (df["sleep_hours"] <= 24)]

# 10. Mostrar nulos depois da limpeza
print("\nNulos depois da limpeza:")
print(df.isnull().sum().sort_values(ascending=False))

# 11. Mostrar tamanho final
print("\nTamanho final do dataset limpo:", df.shape)

# 12. Guardar o dataset limpo
df.to_csv("resultados/dataset_limpo.csv", index=False)

print("\nFicheiro guardado em resultados/dataset_limpo.csv")