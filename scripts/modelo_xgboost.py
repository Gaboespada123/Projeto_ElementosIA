"""
=============================================================================
  SPRINT 3 - MODELOS AVANCADOS: XGBoost + GridSearchCV
  Membro 3 - Comparacao com a baseline do Membro 1
=============================================================================
  Input  : resultados/dados_limpos_final.csv
           resultados/splits_info.pickle
  Output : modelos/modelo_xgb_regressao.pickle
           modelos/modelo_xgb_classificacao.pickle
=============================================================================
"""

# Garantir UTF-8 no terminal Windows (evita UnicodeEncodeError)
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# Compatibilidade com versões antigas do scikit-learn (< 1.4)
try:
    from sklearn.metrics import root_mean_squared_error as _rmse_fn
    def rmse(y_true, y_pred):
        return _rmse_fn(y_true, y_pred)
except ImportError:
    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

# ── Pastas de saída ───────────────────────────────────────────────────────────
os.makedirs("modelos", exist_ok=True)


# =============================================================================
# 1. CARREGAMENTO DO CSV E LIMPEZA DEFENSIVA
#    (mesma lógica do Sprint 2 para garantir consistência entre membros)
# =============================================================================

df_raw = pd.read_csv(os.path.join("resultados", "dados_limpos_final.csv"))

# Normalizar nomes de colunas — remove espaços residuais do OHE do Sprint 1
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))

# Eliminar colunas duplicadas geradas por variantes de maiúsculas no OHE
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated(keep="first")]

# Converter colunas True/False (string) para 0/1
for col in df_raw.columns:
    if df_raw[col].dtype == object:
        unicos = df_raw[col].dropna().str.strip().str.lower().unique()
        if set(unicos).issubset({"true", "false"}):
            df_raw[col] = df_raw[col].str.strip().str.lower().map({"true": 1, "false": 0})

# Forçar dtype numérico e preencher eventuais NaN com a mediana
df_raw = df_raw.apply(pd.to_numeric, errors="coerce")
df_raw = df_raw.fillna(df_raw.median(numeric_only=True))

COLUNA_ALVO = "weight_change_kg_6m"

X_total = df_raw.drop(columns=[COLUNA_ALVO])
y_total = df_raw[COLUNA_ALVO]

print("=" * 66)
print(f"  Dataset carregado : {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas")
print(f"  Features          : {X_total.shape[1]}")
print("=" * 66)


# =============================================================================
# 2. RECRIAR OS SPLITS EXATOS DO MEMBRO 1
#    Carrega o ficheiro pickle com os índices de treino / validação / teste
# =============================================================================

caminho_pickle = os.path.join("resultados", "splits_info.pickle")

with open(caminho_pickle, "rb") as f:
    splits_info = pickle.load(f)

# Suporte a dicionários com chaves em português ou inglês
def _extrair_indice(d, *chaves):
    """Extrai o primeiro valor encontrado de uma lista de chaves candidatas."""
    for chave in chaves:
        if chave in d:
            return d[chave]
    raise KeyError(f"Nenhuma das chaves {chaves} encontrada no pickle. "
                   f"Chaves disponíveis: {list(d.keys())}")

idx_treino = _extrair_indice(splits_info, "indice_treino", "train_index", "idx_train")
idx_val    = _extrair_indice(splits_info, "indice_val",   "val_index",   "idx_val")
idx_teste  = _extrair_indice(splits_info, "indice_teste", "test_index",  "idx_test")

# Recriar os conjuntos com os mesmos índices
X_treino = X_total.iloc[idx_treino].reset_index(drop=True)
X_val    = X_total.iloc[idx_val   ].reset_index(drop=True)
X_teste  = X_total.iloc[idx_teste ].reset_index(drop=True)

y_treino = y_total.iloc[idx_treino].reset_index(drop=True)
y_val    = y_total.iloc[idx_val   ].reset_index(drop=True)
y_teste  = y_total.iloc[idx_teste ].reset_index(drop=True)

print(f"\n  Splits recriados com os índices do Membro 1:")
print(f"    Treino     : {len(X_treino):>5} amostras")
print(f"    Validação  : {len(X_val):>5} amostras")
print(f"    Teste      : {len(X_teste):>5} amostras")


# =============================================================================
# 3. MODELO 1 — XGBoost REGRESSOR
# =============================================================================

print("\n" + "─" * 66)
print("  MODELO 1 — XGBoost Regressor + GridSearchCV")
print("─" * 66)

# -- 3a. Grid de hiperparâmetros a explorar ───────────────────────────────────
#   n_estimators  : nº de árvores (mais = melhor, mas risco de overfitting)
#   max_depth     : profundidade máxima de cada árvore
#   learning_rate : passo de gradiente (eta); valores mais baixos requerem
#                   mais árvores, mas generalizam melhor
param_grid_reg = {
    "n_estimators" : [100, 300, 500],
    "max_depth"    : [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}

xgb_reg_base = XGBRegressor(
    objective        = "reg:squarederror",
    subsample        = 0.8,       # bagging: 80% das amostras por árvore
    colsample_bytree = 0.8,       # 80% das features por árvore
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)

# -- 3b. GridSearchCV — validação cruzada K=5 APENAS no conjunto de treino ────
#   Scoring: neg_root_mean_squared_error (GridSearchCV maximiza → usa negativo)
cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)

grid_reg = GridSearchCV(
    estimator  = xgb_reg_base,
    param_grid = param_grid_reg,
    scoring    = "neg_root_mean_squared_error",
    cv         = cv_reg,
    n_jobs     = -1,
    verbose    = 1,
    refit      = True,   # retreina com os melhores params em todo o treino
)

grid_reg.fit(X_treino, y_treino)

print(f"\n  Melhores hiperparâmetros (Regressão) : {grid_reg.best_params_}")
print(f"  RMSE CV (treino, 5-fold)              : "
      f"{-grid_reg.best_score_:.4f}")

# -- 3c. Melhor modelo já treinado (refit=True) ───────────────────────────────
melhor_reg = grid_reg.best_estimator_

# -- 3d. Métricas nos três conjuntos ──────────────────────────────────────────
metricas_reg = {}
for nome, X_set, y_set in [
    ("Treino",    X_treino, y_treino),
    ("Validação", X_val,    y_val),
    ("Teste",     X_teste,  y_teste),
]:
    y_pred = melhor_reg.predict(X_set)
    metricas_reg[nome] = {
        "RMSE": round(rmse(y_set, y_pred), 4),
        "MAE" : round(mean_absolute_error(y_set, y_pred), 4),
        "R2"  : round(r2_score(y_set, y_pred), 4),
    }

# -- 3e. Guardar modelo em pickle ─────────────────────────────────────────────
caminho_reg = os.path.join("modelos", "modelo_xgb_regressao.pickle")
with open(caminho_reg, "wb") as f:
    pickle.dump(melhor_reg, f)
print(f"\n  [✔] Modelo de regressão guardado → {caminho_reg}")


# =============================================================================
# 4. MODELO 2 — XGBoost CLASSIFIER
# =============================================================================

print("\n" + "─" * 66)
print("  MODELO 2 — XGBoost Classifier + GridSearchCV")
print("─" * 66)

# Definicao binaria: idealmente weight_change < 0 (perdeu peso) = classe 1
# Mas se todos os pacientes perderam peso, a variavel so tem uma classe.
# Nesse caso usamos a mediana como threshold (garante ~50/50).
pct_pos = (y_treino < 0).mean() * 100
if (y_treino < 0).nunique() < 2 or pct_pos > 99 or pct_pos < 1:
    mediana_global = y_total.median()
    y_bin_treino = (y_treino > mediana_global).astype(int)
    y_bin_val    = (y_val    > mediana_global).astype(int)
    y_bin_teste  = (y_teste  > mediana_global).astype(int)
    definicao_bin    = f"weight_change > mediana ({mediana_global:.3f} kg) [fallback]"
    target_names_clf = ["Abaixo da mediana", "Acima da mediana"]
    print(f"\n  [AVISO] weight_change < 0 produz {(y_treino < 0).nunique()} classe(s) ({pct_pos:.1f}% positivo).")
    print(f"  --> Usando fallback: {definicao_bin}")
else:
    mediana_global = y_total.median()
    y_bin_treino = (y_treino < 0).astype(int)
    y_bin_val    = (y_val    < 0).astype(int)
    y_bin_teste  = (y_teste  < 0).astype(int)
    definicao_bin    = "weight_change_kg_6m < 0 (perdeu peso)"
    target_names_clf = ["Insucesso (ganhou peso)", "Sucesso (perdeu peso)"]

print(f"\n  Definicao binaria  : {definicao_bin}")
print(f"  Distribuicao treino: {y_bin_treino.value_counts().to_dict()}")

# -- 4b. Grid de hiperparâmetros ──────────────────────────────────────────────
param_grid_clf = {
    "n_estimators" : [100, 300, 500],
    "max_depth"    : [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}

xgb_clf_base = XGBClassifier(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)

# -- 4c. GridSearchCV — scoring F1 (mais informativo que accuracy em binário) ─
from sklearn.model_selection import StratifiedKFold
# StratifiedKFold requer pelo menos 2 classes -- usar KFold se necessario
if y_bin_treino.nunique() >= 2:
    cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
else:
    cv_clf = KFold(n_splits=5, shuffle=True, random_state=42)
    print("  [AVISO] StratifiedKFold substituido por KFold (target tem 1 classe)")

# Helper: retorna prob da classe positiva de forma segura
def safe_proba(modelo, X):
    proba = modelo.predict_proba(X)
    if proba.shape[1] == 1:
        return proba[:, 0]  # fallback: so ha uma classe
    return proba[:, 1]

def safe_auc(y_true, y_prob):
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)

grid_clf = GridSearchCV(
    estimator  = xgb_clf_base,
    param_grid = param_grid_clf,
    scoring    = "f1",
    cv         = cv_clf,
    n_jobs     = -1,
    verbose    = 1,
    refit      = True,
)

grid_clf.fit(X_treino, y_bin_treino)

print(f"\n  Melhores hiperparâmetros (Classificação) : {grid_clf.best_params_}")
print(f"  F1 CV (treino, 5-fold stratified)         : "
      f"{grid_clf.best_score_:.4f}")

# -- 4d. Melhor modelo e métricas ─────────────────────────────────────────────
melhor_clf = grid_clf.best_estimator_

metricas_clf = {}
for nome, X_set, y_set in [
    ("Treino",    X_treino, y_bin_treino),
    ("Validacao", X_val,    y_bin_val),
    ("Teste",     X_teste,  y_bin_teste),
]:
    y_pred     = melhor_clf.predict(X_set)
    y_prob     = safe_proba(melhor_clf, X_set)
    metricas_clf[nome] = {
        "Accuracy": round(accuracy_score(y_set, y_pred), 4),
        "F1-Score": round(float(f1_score(y_set, y_pred, zero_division=0)), 4),
        "AUC"     : round(safe_auc(y_set, y_prob), 4),
    }

# -- 4e. Guardar modelo ───────────────────────────────────────────────────────
caminho_clf = os.path.join("modelos", "modelo_xgb_classificacao.pickle")
with open(caminho_clf, "wb") as f:
    pickle.dump(melhor_clf, f)
print(f"\n  [✔] Modelo de classificação guardado → {caminho_clf}")


# =============================================================================
# 5. TABELA DE RESUMO — para comparação com a baseline do Membro 1
# =============================================================================

SEP  = "=" * 66
SEP2 = "-" * 66

print(f"\n\n{SEP}")
print("  RESUMO DE MÉTRICAS — SPRINT 3  (XGBoost)  vs  Baseline Membro 1")
print(SEP)

# ── Regressão ─────────────────────────────────────────────────────────────────
print("\n  ▸ MODELO 1: XGBoost Regressor")
print(f"    Melhores hiperparâmetros : {grid_reg.best_params_}")
print(f"\n  {'Conjunto':<14} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
print(f"  {SEP2[:44]}")
for conjunto, vals in metricas_reg.items():
    print(f"  {conjunto:<14} {vals['RMSE']:>10.4f} {vals['MAE']:>10.4f} {vals['R2']:>8.4f}")

# ── Classificação ─────────────────────────────────────────────────────────────
print(f"\n  ▸ MODELO 2: XGBoost Classifier")
print(f"    Melhores hiperparâmetros : {grid_clf.best_params_}")
print(f"\n  {'Conjunto':<14} {'Accuracy':>10} {'F1-Score':>10} {'AUC':>8}")
print(f"  {SEP2[:44]}")
for conjunto, vals in metricas_clf.items():
    print(f"  {conjunto:<14} {vals['Accuracy']:>10.4f} {vals['F1-Score']:>10.4f} {vals['AUC']:>8.4f}")

# ── Nota de comparação ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  COMO COMPARAR COM A BASELINE DO MEMBRO 1:")
print("    Regressão   → compara RMSE e MAE no conjunto de TESTE")
print("    Classificação → compara Accuracy e F1-Score no conjunto de TESTE")
print("    ⚠  Um RMSE de teste >> treino indica overfitting.")
print("    ⚠  Um F1 de validação ≈ teste indica boa generalização.")
print(SEP)
print("\n  Modelos guardados em: modelos/")
print("    • modelo_xgb_regressao.pickle")
print("    • modelo_xgb_classificacao.pickle\n")


# =============================================================================
# 6. BÓNUS: IMPORTÂNCIA DAS VARIÁVEIS (FEATURE IMPORTANCE)
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

print(f"\n{SEP}")
print("  EXTRAINDO A IMPORTÂNCIA DAS VARIÁVEIS (XGBoost Classifier)")
print(SEP)

# 1. Obter a importância das features do melhor modelo de classificação
importancias = melhor_clf.feature_importances_
nomes_features = X_treino.columns

# 2. Criar um DataFrame e ordenar as top 10 variáveis mais importantes
df_importancia = pd.DataFrame({
    'Feature': nomes_features,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False).head(10)

print("\n  Top 10 Variáveis que mais influenciam o Sucesso da Dieta:")
print(df_importancia.to_string(index=False))

# 3. Gerar e guardar um gráfico elegante para a apresentação (Membro 2) e Relatório (Tu)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Feature', data=df_importancia, palette='viridis')
plt.title('Top 10 Variáveis Mais Importantes (XGBoost)', fontsize=14, fontweight='bold')
plt.xlabel('Importância Relativa', fontsize=12)
plt.ylabel('Variável', fontsize=12)
plt.tight_layout()

# Guardar na pasta de gráficos
caminho_grafico_importancia = os.path.join("resultados", "graficos", "xgboost_feature_importance.png")
plt.savefig(caminho_grafico_importancia, dpi=150)
plt.close()

print(f"\n  [✔] Gráfico de Importância guardado → {caminho_grafico_importancia}\n")