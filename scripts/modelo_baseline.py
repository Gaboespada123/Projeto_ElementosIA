import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, classification_report
import pickle
import os

# ==================== CARREGAR DADOS ====================

print("Carregando dataset...")
df = pd.read_csv("resultados/dados_limpos_final.csv")

print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

# ==================== PREPARAR VARIÁVEIS ====================

coluna_alvo = "weight_change_kg_6m"

if coluna_alvo not in df.columns:
    print(f"Erro: Coluna '{coluna_alvo}' não encontrada.")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    exit()

X = df.drop(columns=[coluna_alvo])
y = df[coluna_alvo]

print(f"Características: {X.shape}")
print(f"Variável alvo: {y.shape}")

# ==================== CRIAR SPLITS ====================

print("\nCriando splits (50% treino, 30% validação, 10% teste, 10% não utilizado)...")

X_trabalho, X_nao_utilizado, y_trabalho, y_nao_utilizado = train_test_split(
    X, y, test_size=0.10, random_state=42
)

X_treino, X_temp, y_treino, y_temp = train_test_split(
    X_trabalho, y_trabalho, train_size=0.5556, random_state=42
)

X_val, X_teste, y_val, y_teste = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print(f"Treino: {X_treino.shape[0]} amostras ({X_treino.shape[0]/X.shape[0]*100:.1f}%)")
print(f"Validação: {X_val.shape[0]} amostras ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
print(f"Teste: {X_teste.shape[0]} amostras ({X_teste.shape[0]/X.shape[0]*100:.1f}%)")
print(f"Não utilizado: {X_nao_utilizado.shape[0]} amostras ({X_nao_utilizado.shape[0]/X.shape[0]*100:.1f}%)")

# Guardar índices dos splits para uso posterior
informacao_splits = {
    "indice_treino": X_treino.index.tolist(),
    "indice_val": X_val.index.tolist(),
    "indice_teste": X_teste.index.tolist(),
    "indice_nao_utilizado": X_nao_utilizado.index.tolist()
}

with open("resultados/splits_info.pickle", "wb") as f:
    pickle.dump(informacao_splits, f)

print("Splits guardados em 'resultados/splits_info.pickle'")

# ==================== MODELO 1: REGRESSÃO LINEAR ====================

print("\n" + "="*60)
print("MODELO 1: REGRESSÃO LINEAR")
print("="*60)

modelo_rl = LinearRegression()
modelo_rl.fit(X_treino, y_treino)

y_treino_pred = modelo_rl.predict(X_treino)
y_val_pred = modelo_rl.predict(X_val)
y_teste_pred = modelo_rl.predict(X_teste)

rmse_treino = np.sqrt(mean_squared_error(y_treino, y_treino_pred))
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
rmse_teste = np.sqrt(mean_squared_error(y_teste, y_teste_pred))

mae_treino = mean_absolute_error(y_treino, y_treino_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
mae_teste = mean_absolute_error(y_teste, y_teste_pred)

print(f"\nTreino - RMSE: {rmse_treino:.4f} | MAE: {mae_treino:.4f}")
print(f"Validação - RMSE: {rmse_val:.4f} | MAE: {mae_val:.4f}")
print(f"Teste - RMSE: {rmse_teste:.4f} | MAE: {mae_teste:.4f}")

importancia_features_rl = pd.DataFrame({
    'Característica': X_treino.columns,
    'Coeficiente': modelo_rl.coef_
}).sort_values('Coeficiente', key=abs, ascending=False)

print("\nTop 5 características mais importantes:")
print(importancia_features_rl.head().to_string(index=False))

with open("modelos/modelo_regressao_linear.pickle", "wb") as f:
    pickle.dump(modelo_rl, f)

print("\nModelo guardado: 'modelos/modelo_regressao_linear.pickle'")

# ==================== MODELO 2: REGRESSÃO LOGÍSTICA ====================

print("\n" + "="*60)
print("MODELO 2: REGRESSÃO LOGÍSTICA")
print("="*60)

y_binaria = (y > y.median()).astype(int)

print(f"\nDistribuição das classes:")
print(y_binaria.value_counts())
print(f"Equilíbrio: {y_binaria.value_counts()[1]/len(y_binaria)*100:.1f}% classe positiva")

y_treino_bin = y_binaria.iloc[X_treino.index]
y_val_bin = y_binaria.iloc[X_val.index]
y_teste_bin = y_binaria.iloc[X_teste.index]

modelo_rl_logistica = LogisticRegression(max_iter=1000, random_state=42)
modelo_rl_logistica.fit(X_treino, y_treino_bin)

y_treino_pred_bin = modelo_rl_logistica.predict(X_treino)
y_val_pred_bin = modelo_rl_logistica.predict(X_val)
y_teste_pred_bin = modelo_rl_logistica.predict(X_teste)

acc_treino = accuracy_score(y_treino_bin, y_treino_pred_bin)
acc_val = accuracy_score(y_val_bin, y_val_pred_bin)
acc_teste = accuracy_score(y_teste_bin, y_teste_pred_bin)

f1_treino = f1_score(y_treino_bin, y_treino_pred_bin)
f1_val = f1_score(y_val_bin, y_val_pred_bin)
f1_teste = f1_score(y_teste_bin, y_teste_pred_bin)

print(f"\nTreino - Acurácia: {acc_treino:.4f} | F1-Score: {f1_treino:.4f}")
print(f"Validação - Acurácia: {acc_val:.4f} | F1-Score: {f1_val:.4f}")
print(f"Teste - Acurácia: {acc_teste:.4f} | F1-Score: {f1_teste:.4f}")

print(f"\nRelatório de Classificação (Conjunto de Teste):")
print(classification_report(y_teste_bin, y_teste_pred_bin, target_names=["Sem Sucesso", "Com Sucesso"]))

with open("modelos/modelo_regressao_logistica.pickle", "wb") as f:
    pickle.dump(modelo_rl_logistica, f)

print("Modelo guardado: 'modelos/modelo_regressao_logistica.pickle'")

# ==================== RESUMO DE RESULTADOS ====================

print("\n" + "="*60)
print("RESUMO DE RESULTADOS")
print("="*60)

resultados = pd.DataFrame({
    'Modelo': ['Regressão Linear', 'Regressão Logística'],
    'Treino (RMSE/Acurácia)': [f"{rmse_treino:.4f}", f"{acc_treino:.4f}"],
    'Validação (RMSE/Acurácia)': [f"{rmse_val:.4f}", f"{acc_val:.4f}"],
    'Teste (RMSE/Acurácia)': [f"{rmse_teste:.4f}", f"{acc_teste:.4f}"]
})

print("\n" + resultados.to_string(index=False))

resultados.to_csv("resultados/resultados_modelos_baseline.csv", index=False)
print("\nResultados guardados: 'resultados/resultados_modelos_baseline.csv'")

print("\n" + "="*60)
print("Treinamento concluído")
print("="*60)