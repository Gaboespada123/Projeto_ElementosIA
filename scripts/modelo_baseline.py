import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, classification_report, roc_auc_score
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

r2_treino = r2_score(y_treino, y_treino_pred)
r2_val = r2_score(y_val, y_val_pred)
r2_teste = r2_score(y_teste, y_teste_pred)

print(f"\nTreino    - RMSE: {rmse_treino:.4f} | MAE: {mae_treino:.4f} | R²: {r2_treino:.4f}")
print(f"Validação - RMSE: {rmse_val:.4f} | MAE: {mae_val:.4f} | R²: {r2_val:.4f}")
print(f"Teste     - RMSE: {rmse_teste:.4f} | MAE: {mae_teste:.4f} | R²: {r2_teste:.4f}")

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

# Definição de sucesso: perda de peso (weight_change_kg_6m < 0) = classe 1
# Consistente com a definição usada na AED (01_AED_Univariada.py)
y_binaria = (y < 0).astype(int)

n_classes  = y_binaria.nunique()
pct_classe1 = y_binaria.mean() * 100

print(f"\nDefinição de sucesso: weight_change_kg_6m < 0 (perdeu peso)")
print(f"Distribuição das classes:")
print(y_binaria.value_counts().rename({1: 'Sucesso (perdeu peso)', 0: 'Insucesso'}))
print(f"Equilíbrio: {pct_classe1:.1f}% de sucesso (perda de peso)")

# Deteção automática: se só existe uma classe, usar mediana como threshold alternativo.
# Isto acontece quando todos (ou nenhum) os pacientes perderam peso.
# A divisão pela mediana garante ~50/50 e é válida academicamente para separar
# "perdeu mais do que a mediana" vs "perdeu menos/ganhou".
if n_classes < 2 or pct_classe1 > 99 or pct_classe1 < 1:
    print(f"\n[AVISO] 'weight_change < 0' produz {n_classes} classe(s) ({pct_classe1:.1f}% positivo).")
    print("  O classificador requer pelo menos 2 classes para treinar.")
    print("  --> Usando threshold alternativo: weight_change_kg_6m > mediana")
    print(f"    (mediana = {y.median():.4f} kg | classe 1 = perdeu mais do que a mediana)")
    mediana_global  = y.median()
    y_binaria       = (y > mediana_global).astype(int)
    definicao_bin   = f"weight_change_kg_6m > mediana ({mediana_global:.3f} kg)"
    target_names_clf = ["Abaixo da mediana", "Acima da mediana"]
else:
    mediana_global   = y.median()
    definicao_bin    = "weight_change_kg_6m < 0 (perdeu peso)"
    target_names_clf = ["Insucesso (ganhou peso)", "Sucesso (perdeu peso)"]

print(f"\n  Definicao final usada   : {definicao_bin}")
print(f"  Distribuicao final      : {y_binaria.value_counts().to_dict()}")

# Usar .loc para garantir alinhamento correto de indices
y_treino_bin = y_binaria.loc[X_treino.index]
y_val_bin    = y_binaria.loc[X_val.index]
y_teste_bin  = y_binaria.loc[X_teste.index]

# Verificacao de seguranca: confirmar que todos os splits tem ambas as classes
for nome_split, y_split in [("Treino", y_treino_bin), ("Validacao", y_val_bin), ("Teste", y_teste_bin)]:
    if y_split.nunique() < 2:
        print(f"  [AVISO] {nome_split} so tem uma classe: {y_split.unique()}")

# class_weight="balanced" torna o modelo robusto a possíveis desbalanceamentos
modelo_rl_logistica = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
modelo_rl_logistica.fit(X_treino, y_treino_bin)

y_treino_pred_bin = modelo_rl_logistica.predict(X_treino)
y_val_pred_bin    = modelo_rl_logistica.predict(X_val)
y_teste_pred_bin  = modelo_rl_logistica.predict(X_teste)

acc_treino = accuracy_score(y_treino_bin, y_treino_pred_bin)
acc_val    = accuracy_score(y_val_bin, y_val_pred_bin)
acc_teste  = accuracy_score(y_teste_bin, y_teste_pred_bin)

f1_treino = f1_score(y_treino_bin, y_treino_pred_bin, zero_division=0)
f1_val    = f1_score(y_val_bin, y_val_pred_bin, zero_division=0)
f1_teste  = f1_score(y_teste_bin, y_teste_pred_bin, zero_division=0)

# AUC: proteção extra caso um split só tenha uma classe
def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)

auc_treino = safe_auc(y_treino_bin, modelo_rl_logistica.predict_proba(X_treino)[:, 1])
auc_val    = safe_auc(y_val_bin,    modelo_rl_logistica.predict_proba(X_val)[:, 1])
auc_teste  = safe_auc(y_teste_bin,  modelo_rl_logistica.predict_proba(X_teste)[:, 1])

print(f"\nTreino    - Acurácia: {acc_treino:.4f} | F1-Score: {f1_treino:.4f} | AUC: {auc_treino:.4f}")
print(f"Validação - Acurácia: {acc_val:.4f} | F1-Score: {f1_val:.4f} | AUC: {auc_val:.4f}")
print(f"Teste     - Acurácia: {acc_teste:.4f} | F1-Score: {f1_teste:.4f} | AUC: {auc_teste:.4f}")

print(f"\nRelatório de Classificação (Conjunto de Teste):")
print(classification_report(y_teste_bin, y_teste_pred_bin, target_names=target_names_clf, zero_division=0))

with open("modelos/modelo_regressao_logistica.pickle", "wb") as f:
    pickle.dump(modelo_rl_logistica, f)

print("Modelo guardado: 'modelos/modelo_regressao_logistica.pickle'")

# ==================== RESUMO DE RESULTADOS ====================

print("\n" + "="*60)
print("RESUMO DE RESULTADOS")
print("="*60)

resultados = pd.DataFrame({
    'Modelo'    : ['Regressão Linear', 'Regressão Logística'],
    'Tarefa'    : ['Regressão',        'Classificação'],
    'Métrica'   : ['RMSE | MAE | R²',  'Accuracy | F1 | AUC'],
    'Treino'    : [f"{rmse_treino:.4f} | {mae_treino:.4f} | {r2_treino:.4f}",
                   f"{acc_treino:.4f} | {f1_treino:.4f} | {auc_treino:.4f}"],
    'Validação' : [f"{rmse_val:.4f} | {mae_val:.4f} | {r2_val:.4f}",
                   f"{acc_val:.4f} | {f1_val:.4f} | {auc_val:.4f}"],
    'Teste'     : [f"{rmse_teste:.4f} | {mae_teste:.4f} | {r2_teste:.4f}",
                   f"{acc_teste:.4f} | {f1_teste:.4f} | {auc_teste:.4f}"],
})

print("\n" + resultados.to_string(index=False))

resultados.to_csv("resultados/resultados_modelos_baseline.csv", index=False)
print("\nResultados guardados: 'resultados/resultados_modelos_baseline.csv'")

print("\n" + "="*60)
print("Treinamento concluído")
print("="*60)