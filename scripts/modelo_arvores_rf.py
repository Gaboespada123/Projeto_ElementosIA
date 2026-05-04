"""
=============================================================================
  SPRINT 3 — MODELOS MEMBRO 2: Árvores de Decisão + Random Forest
  Feature Importance: que variáveis do nutricionista ou do paciente
  mais influenciam o resultado?
=============================================================================
  Input  : resultados/dados_limpos_final.csv
           resultados/splits_info.pickle
  Output : modelos/modelo_arvore_regressao.pickle
           modelos/modelo_arvore_classificacao.pickle
           modelos/modelo_rf_regressao.pickle
           modelos/modelo_rf_classificacao.pickle
           resultados/graficos/rf_feature_importance_regressao.png
           resultados/graficos/rf_feature_importance_classificacao.png
           resultados/graficos/comparacao_feature_importance.png
           resultados/graficos/feature_importance_por_categoria.png
           resultados/graficos/arvore_decisao_visualizacao.png
           resultados/resultados_modelos_arvores_rf.csv
=============================================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    plot_tree,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# Compatibilidade com sklearn < 1.4
try:
    from sklearn.metrics import root_mean_squared_error as _rmse_fn
    def rmse(y_true, y_pred):
        return _rmse_fn(y_true, y_pred)
except ImportError:
    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


# ── Pastas de saída ────────────────────────────────────────────────────────────
os.makedirs("modelos", exist_ok=True)
os.makedirs(os.path.join("resultados", "graficos"), exist_ok=True)

GRAFICOS = os.path.join("resultados", "graficos")
SEP  = "=" * 66
SEP2 = "-" * 66


# =============================================================================
# 1. CARREGAMENTO DO CSV E LIMPEZA DEFENSIVA
#    (mesma lógica do Membro 3 para garantir consistência entre membros)
# =============================================================================

print(SEP)
print("  SPRINT 3 — Árvores de Decisão + Random Forest  (Membro 2)")
print(SEP)

df_raw = pd.read_csv(os.path.join("resultados", "dados_limpos_final.csv"))

# Normalizar nomes de colunas — remove espaços residuais do OHE do Sprint 1
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated(keep="first")]

# Converter colunas True/False em string para 0/1
for col in df_raw.columns:
    if df_raw[col].dtype == object:
        unicos = df_raw[col].dropna().str.strip().str.lower().unique()
        if set(unicos).issubset({"true", "false"}):
            df_raw[col] = df_raw[col].str.strip().str.lower().map({"true": 1, "false": 0})

df_raw = df_raw.apply(pd.to_numeric, errors="coerce")
df_raw = df_raw.fillna(df_raw.median(numeric_only=True))

COLUNA_ALVO = "weight_change_kg_6m"
X_total = df_raw.drop(columns=[COLUNA_ALVO])
y_total = df_raw[COLUNA_ALVO]

print(f"\n  Dataset carregado : {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas")
print(f"  Features          : {X_total.shape[1]}")


# =============================================================================
# 2. RECRIAR OS SPLITS EXATOS DO MEMBRO 1
# =============================================================================

def _extrair_indice(d, *chaves):
    for chave in chaves:
        if chave in d:
            return d[chave]
    raise KeyError(
        f"Nenhuma das chaves {chaves} encontrada. Chaves disponíveis: {list(d.keys())}"
    )


with open(os.path.join("resultados", "splits_info.pickle"), "rb") as f:
    splits_info = pickle.load(f)

idx_treino = _extrair_indice(splits_info, "indice_treino", "train_index", "idx_train")
idx_val    = _extrair_indice(splits_info, "indice_val",   "val_index",   "idx_val")
idx_teste  = _extrair_indice(splits_info, "indice_teste", "test_index",  "idx_test")

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

# Variável binária — consistente com Membro 1 e Membro 3
# Classe 1 → acima da mediana (maior variação de peso)
# Classe 0 → abaixo ou igual à mediana
mediana_global = y_total.median()
y_bin_treino = (y_treino > mediana_global).astype(int)
y_bin_val    = (y_val    > mediana_global).astype(int)
y_bin_teste  = (y_teste  > mediana_global).astype(int)

print(f"\n  Mediana global de y : {mediana_global:.4f}")
print(f"  Distribuição treino — Classe 1: {y_bin_treino.mean()*100:.1f}%  "
      f"| Classe 0: {(1 - y_bin_treino.mean())*100:.1f}%")


# Validadores cruzados partilhados
cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =============================================================================
# 3. ÁRVORE DE DECISÃO — REGRESSÃO
# =============================================================================

print("\n" + SEP2)
print("  MODELO 1 — Árvore de Decisão (Regressão)")
print(SEP2)

param_dt = {
    "max_depth"        : [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf" : [1, 2, 5, 10],
}

grid_dt_reg = GridSearchCV(
    estimator  = DecisionTreeRegressor(random_state=42),
    param_grid = param_dt,
    scoring    = "neg_root_mean_squared_error",
    cv         = cv_reg,
    n_jobs     = -1,
    verbose    = 0,
    refit      = True,
)
grid_dt_reg.fit(X_treino, y_treino)
melhor_dt_reg = grid_dt_reg.best_estimator_

print(f"\n  Melhores hiperparâmetros : {grid_dt_reg.best_params_}")
print(f"  RMSE CV (5-fold)         : {-grid_dt_reg.best_score_:.4f}")

metricas_dt_reg = {}
for nome, X_set, y_set in [
    ("Treino",    X_treino, y_treino),
    ("Validação", X_val,    y_val),
    ("Teste",     X_teste,  y_teste),
]:
    pred = melhor_dt_reg.predict(X_set)
    metricas_dt_reg[nome] = {
        "RMSE": round(rmse(y_set, pred), 4),
        "MAE" : round(mean_absolute_error(y_set, pred), 4),
    }
    print(f"  {nome:<12}  RMSE: {metricas_dt_reg[nome]['RMSE']:.4f}  |  "
          f"MAE: {metricas_dt_reg[nome]['MAE']:.4f}")

with open(os.path.join("modelos", "modelo_arvore_regressao.pickle"), "wb") as f:
    pickle.dump(melhor_dt_reg, f)
print("  [✔] Modelo guardado → modelos/modelo_arvore_regressao.pickle")


# =============================================================================
# 4. ÁRVORE DE DECISÃO — CLASSIFICAÇÃO
# =============================================================================

print("\n" + SEP2)
print("  MODELO 2 — Árvore de Decisão (Classificação)")
print(SEP2)

grid_dt_clf = GridSearchCV(
    estimator  = DecisionTreeClassifier(random_state=42),
    param_grid = param_dt,
    scoring    = "f1",
    cv         = cv_clf,
    n_jobs     = -1,
    verbose    = 0,
    refit      = True,
)
grid_dt_clf.fit(X_treino, y_bin_treino)
melhor_dt_clf = grid_dt_clf.best_estimator_

print(f"\n  Melhores hiperparâmetros : {grid_dt_clf.best_params_}")
print(f"  F1 CV (5-fold stratified) : {grid_dt_clf.best_score_:.4f}")

metricas_dt_clf = {}
for nome, X_set, y_set in [
    ("Treino",    X_treino, y_bin_treino),
    ("Validação", X_val,    y_bin_val),
    ("Teste",     X_teste,  y_bin_teste),
]:
    pred = melhor_dt_clf.predict(X_set)
    metricas_dt_clf[nome] = {
        "Accuracy": round(accuracy_score(y_set, pred), 4),
        "F1-Score": round(float(f1_score(y_set, pred, zero_division=0)), 4),
    }
    print(f"  {nome:<12}  Accuracy: {metricas_dt_clf[nome]['Accuracy']:.4f}  |  "
          f"F1: {metricas_dt_clf[nome]['F1-Score']:.4f}")

print(f"\n  Relatório de Classificação (Conjunto de Teste):")
print(classification_report(
    y_bin_teste,
    melhor_dt_clf.predict(X_teste),
    target_names=["Abaixo da mediana", "Acima da mediana"],
))

with open(os.path.join("modelos", "modelo_arvore_classificacao.pickle"), "wb") as f:
    pickle.dump(melhor_dt_clf, f)
print("  [✔] Modelo guardado → modelos/modelo_arvore_classificacao.pickle")


# =============================================================================
# 5. RANDOM FOREST — REGRESSÃO
# =============================================================================

print("\n" + SEP2)
print("  MODELO 3 — Random Forest (Regressão)")
print(SEP2)

param_rf = {
    "n_estimators"     : [100, 200, 300],
    "max_depth"        : [5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
}

grid_rf_reg = GridSearchCV(
    estimator  = RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid = param_rf,
    scoring    = "neg_root_mean_squared_error",
    cv         = cv_reg,
    n_jobs     = -1,
    verbose    = 1,
    refit      = True,
)
grid_rf_reg.fit(X_treino, y_treino)
melhor_rf_reg = grid_rf_reg.best_estimator_

print(f"\n  Melhores hiperparâmetros : {grid_rf_reg.best_params_}")
print(f"  RMSE CV (5-fold)         : {-grid_rf_reg.best_score_:.4f}")

metricas_rf_reg = {}
for nome, X_set, y_set in [
    ("Treino",    X_treino, y_treino),
    ("Validação", X_val,    y_val),
    ("Teste",     X_teste,  y_teste),
]:
    pred = melhor_rf_reg.predict(X_set)
    metricas_rf_reg[nome] = {
        "RMSE": round(rmse(y_set, pred), 4),
        "MAE" : round(mean_absolute_error(y_set, pred), 4),
    }
    print(f"  {nome:<12}  RMSE: {metricas_rf_reg[nome]['RMSE']:.4f}  |  "
          f"MAE: {metricas_rf_reg[nome]['MAE']:.4f}")

with open(os.path.join("modelos", "modelo_rf_regressao.pickle"), "wb") as f:
    pickle.dump(melhor_rf_reg, f)
print("  [✔] Modelo guardado → modelos/modelo_rf_regressao.pickle")


# =============================================================================
# 6. RANDOM FOREST — CLASSIFICAÇÃO
# =============================================================================

print("\n" + SEP2)
print("  MODELO 4 — Random Forest (Classificação)")
print(SEP2)

grid_rf_clf = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid = param_rf,
    scoring    = "f1",
    cv         = cv_clf,
    n_jobs     = -1,
    verbose    = 1,
    refit      = True,
)
grid_rf_clf.fit(X_treino, y_bin_treino)
melhor_rf_clf = grid_rf_clf.best_estimator_

print(f"\n  Melhores hiperparâmetros : {grid_rf_clf.best_params_}")
print(f"  F1 CV (5-fold stratified) : {grid_rf_clf.best_score_:.4f}")

metricas_rf_clf = {}
for nome, X_set, y_set in [
    ("Treino",    X_treino, y_bin_treino),
    ("Validação", X_val,    y_bin_val),
    ("Teste",     X_teste,  y_bin_teste),
]:
    pred = melhor_rf_clf.predict(X_set)
    metricas_rf_clf[nome] = {
        "Accuracy": round(accuracy_score(y_set, pred), 4),
        "F1-Score": round(float(f1_score(y_set, pred, zero_division=0)), 4),
    }
    print(f"  {nome:<12}  Accuracy: {metricas_rf_clf[nome]['Accuracy']:.4f}  |  "
          f"F1: {metricas_rf_clf[nome]['F1-Score']:.4f}")

print(f"\n  Relatório de Classificação (Conjunto de Teste):")
print(classification_report(
    y_bin_teste,
    melhor_rf_clf.predict(X_teste),
    target_names=["Abaixo da mediana", "Acima da mediana"],
))

with open(os.path.join("modelos", "modelo_rf_classificacao.pickle"), "wb") as f:
    pickle.dump(melhor_rf_clf, f)
print("  [✔] Modelo guardado → modelos/modelo_rf_classificacao.pickle")


# =============================================================================
# 7. FEATURE IMPORTANCE — ANÁLISE COMPLETA
#    Responde à questão: variáveis do nutricionista, do paciente ou da dieta
#    são as que mais influenciam a variação de peso?
# =============================================================================

print("\n" + SEP2)
print("  ANÁLISE DE FEATURE IMPORTANCE")
print(SEP2)

feature_names = X_total.columns.tolist()
TOP_N = 20


def extrair_importancias(modelo, nomes, top_n=TOP_N):
    df_fi = pd.DataFrame({
        "Feature"    : nomes,
        "Importância": modelo.feature_importances_,
    }).sort_values("Importância", ascending=False).head(top_n).reset_index(drop=True)
    df_fi["Rank"] = df_fi.index + 1
    return df_fi


fi_rf_reg = extrair_importancias(melhor_rf_reg, feature_names)
fi_rf_clf = extrair_importancias(melhor_rf_clf, feature_names)

print(f"\n  Top 10 features — RF Regressor:")
for _, row in fi_rf_reg.head(10).iterrows():
    bar = "█" * max(1, int(row["Importância"] * 120))
    print(f"  {row['Rank']:>2}. {row['Feature']:<35} {row['Importância']:.4f}  {bar}")

print(f"\n  Top 10 features — RF Classifier:")
for _, row in fi_rf_clf.head(10).iterrows():
    bar = "█" * max(1, int(row["Importância"] * 120))
    print(f"  {row['Rank']:>2}. {row['Feature']:<35} {row['Importância']:.4f}  {bar}")


# ── Categorização das features por origem ─────────────────────────────────────
# Permite responder à pergunta de negócio do Sprint 3
CATEGORIAS = {
    "Paciente"     : ["age", "height", "weight", "bmi", "sleep", "sex",
                      "motivation_score", "physical"],
    "Nutricionista": ["years_experience", "specialization", "approach"],
    "Dieta"        : ["calorie", "protein", "carb", "fat", "fiber",
                      "sodium", "diet", "macro"],
    "Programa"     : ["adherence", "weeks", "motivation_score_program",
                      "season", "program"],
}

def classificar_feature(nome):
    nome_lower = nome.lower()
    for cat, palavras_chave in CATEGORIAS.items():
        if any(p in nome_lower for p in palavras_chave):
            return cat
    return "Outro"

# Construir DataFrame completo de importâncias (todos os features)
df_fi_completo = pd.DataFrame({
    "Feature"       : feature_names,
    "RF_Reg"        : melhor_rf_reg.feature_importances_,
    "RF_Clf"        : melhor_rf_clf.feature_importances_,
    "DT_Reg"        : melhor_dt_reg.feature_importances_,
    "DT_Clf"        : melhor_dt_clf.feature_importances_,
})
df_fi_completo["Categoria"] = df_fi_completo["Feature"].apply(classificar_feature)

# Importância agregada por categoria (soma das importâncias das features de cada grupo)
fi_por_cat_reg = (
    df_fi_completo.groupby("Categoria")["RF_Reg"].sum()
    .sort_values(ascending=False)
)
fi_por_cat_clf = (
    df_fi_completo.groupby("Categoria")["RF_Clf"].sum()
    .sort_values(ascending=False)
)

print(f"\n  Importância agregada por categoria — RF Regressor:")
for cat, val in fi_por_cat_reg.items():
    bar = "█" * max(1, int(val * 100))
    print(f"    {cat:<16} {val:.4f}  {bar}")

print(f"\n  Importância agregada por categoria — RF Classifier:")
for cat, val in fi_por_cat_clf.items():
    bar = "█" * max(1, int(val * 100))
    print(f"    {cat:<16} {val:.4f}  {bar}")


# =============================================================================
# 8. GRÁFICOS
# =============================================================================

# ─── Gráfico 1: Feature Importance RF Regressor ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
cores = plt.colormaps["RdYlGn"](np.linspace(0.25, 0.9, len(fi_rf_reg)))
bars = ax.barh(
    fi_rf_reg["Feature"].iloc[::-1],
    fi_rf_reg["Importância"].iloc[::-1],
    color=cores,
)
for bar, val in zip(bars, fi_rf_reg["Importância"].iloc[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=8)
ax.set_xlabel("Importância (Mean Decrease Impurity)", fontsize=11)
ax.set_title(
    "Random Forest Regressor\nTop Features — Previsão de Variação de Peso (kg)",
    fontsize=13, fontweight="bold",
)
ax.tick_params(axis="both", labelsize=9)
plt.tight_layout()
caminho = os.path.join(GRAFICOS, "rf_feature_importance_regressao.png")
plt.savefig(caminho, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  [✔] Gráfico guardado → {caminho}")


# ─── Gráfico 2: Feature Importance RF Classifier ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
cores = plt.colormaps["RdYlBu"](np.linspace(0.15, 0.85, len(fi_rf_clf)))
bars = ax.barh(
    fi_rf_clf["Feature"].iloc[::-1],
    fi_rf_clf["Importância"].iloc[::-1],
    color=cores,
)
for bar, val in zip(bars, fi_rf_clf["Importância"].iloc[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=8)
ax.set_xlabel("Importância (Mean Decrease Impurity)", fontsize=11)
ax.set_title(
    "Random Forest Classifier\nTop Features — Classificação Acima/Abaixo da Mediana",
    fontsize=13, fontweight="bold",
)
ax.tick_params(axis="both", labelsize=9)
plt.tight_layout()
caminho = os.path.join(GRAFICOS, "rf_feature_importance_classificacao.png")
plt.savefig(caminho, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [✔] Gráfico guardado → {caminho}")


# ─── Gráfico 3: Comparação RF Reg vs RF Clf (top features da união) ───────────
top15_reg = fi_rf_reg.head(15)["Feature"].tolist()
top15_clf = fi_rf_clf.head(15)["Feature"].tolist()
features_uniao = list(dict.fromkeys(top15_reg + top15_clf))

df_comp = df_fi_completo.set_index("Feature").loc[features_uniao, ["RF_Reg", "RF_Clf"]]
df_comp = df_comp.sort_values("RF_Reg", ascending=True)

y_pos  = np.arange(len(df_comp))
altura = 0.38
fig, ax = plt.subplots(figsize=(12, max(8, len(df_comp) * 0.48)))
ax.barh(y_pos + altura / 2, df_comp["RF_Reg"].to_numpy(), altura,
        label="RF Regressor (RMSE)", color="#27ae60", alpha=0.85)
ax.barh(y_pos - altura / 2, df_comp["RF_Clf"].to_numpy(), altura,
        label="RF Classifier (F1)",  color="#2980b9", alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(df_comp.index, fontsize=9)
ax.set_xlabel("Importância (Mean Decrease Impurity)", fontsize=11)
ax.set_title(
    "Comparação de Feature Importance\nRandom Forest: Regressão vs Classificação",
    fontsize=13, fontweight="bold",
)
ax.legend(fontsize=10)
ax.tick_params(axis="x", labelsize=9)
plt.tight_layout()
caminho = os.path.join(GRAFICOS, "comparacao_feature_importance.png")
plt.savefig(caminho, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [✔] Gráfico guardado → {caminho}")


# ─── Gráfico 4: Importância por categoria (negócio) ───────────────────────────
# Este gráfico responde diretamente à questão do Sprint 3:
# "variáveis do nutricionista ou do paciente mandam mais no modelo?"
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cores_cat = {
    "Paciente"     : "#e74c3c",
    "Dieta"        : "#f39c12",
    "Programa"     : "#27ae60",
    "Nutricionista": "#2980b9",
    "Outro"        : "#95a5a6",
}

for ax, (fi_cat, titulo) in zip(
    axes,
    [
        (fi_por_cat_reg, "RF Regressor\n(Variação de Peso em kg)"),
        (fi_por_cat_clf, "RF Classifier\n(Acima/Abaixo da Mediana)"),
    ],
):
    categorias_ord = fi_cat.index.tolist()
    valores        = fi_cat.values
    cores_ord      = [cores_cat.get(c, "#95a5a6") for c in categorias_ord]
    barras         = ax.bar(categorias_ord, valores, color=cores_ord, edgecolor="white",
                            linewidth=1.2, width=0.6)
    for barra, val in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2, val + 0.005,
                f"{val:.3f}\n({val*100:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(valores) * 1.25)
    ax.set_ylabel("Importância total (soma)", fontsize=10)
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "Feature Importance por Categoria de Origem\n"
    "Qual grupo de variáveis mais influencia o resultado do programa?",
    fontsize=13, fontweight="bold", y=1.02,
)
plt.tight_layout()
caminho = os.path.join(GRAFICOS, "feature_importance_por_categoria.png")
plt.savefig(caminho, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [✔] Gráfico guardado → {caminho}")


# ─── Gráfico 5: Visualização da Árvore de Decisão ─────────────────────────────
# Para legibilidade visual usa-se profundidade=4.
# O modelo de produção (melhor_dt_clf) é diferente — este serve apenas para ilustrar
# as regras de decisão aprendidas.
dt_visual = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=max(5, len(X_treino) // 50),
    random_state=42,
)
dt_visual.fit(X_treino, y_bin_treino)
acc_dt_visual = accuracy_score(y_bin_treino, dt_visual.predict(X_treino))

nomes_limpos = [f.replace("_", " ").title() for f in feature_names]

fig, ax = plt.subplots(figsize=(26, 12))
plot_tree(
    dt_visual,
    feature_names=nomes_limpos,
    class_names=["Abaixo Mediana", "Acima Mediana"],
    filled=True,
    rounded=True,
    fontsize=7,
    ax=ax,
    proportion=False,
    impurity=True,
    precision=3,
)
ax.set_title(
    f"Árvore de Decisão (profundidade=4, para visualização)\n"
    f"Accuracy treino: {acc_dt_visual:.4f}  |  "
    f"As caixas verdes indicam maioria 'Acima da Mediana'; "
    f"laranja = 'Abaixo da Mediana'",
    fontsize=12, fontweight="bold", pad=12,
)
plt.tight_layout()
caminho = os.path.join(GRAFICOS, "arvore_decisao_visualizacao.png")
plt.savefig(caminho, dpi=110, bbox_inches="tight")
plt.close()
print(f"  [✔] Gráfico guardado → {caminho}")


# =============================================================================
# 9. TABELA RESUMO E CSV
# =============================================================================

print(f"\n\n{SEP}")
print("  RESUMO DE MÉTRICAS — Árvore de Decisão + Random Forest")
print(SEP)

print(f"\n  ▸ REGRESSÃO (RMSE / MAE)")
print(f"  {'Modelo':<28} {'Conjunto':<12} {'RMSE':>8} {'MAE':>8}")
print(f"  {SEP2[:58]}")
for nome_modelo, metricas in [
    ("Árvore de Decisão", metricas_dt_reg),
    ("Random Forest",     metricas_rf_reg),
]:
    for conjunto, vals in metricas.items():
        print(f"  {nome_modelo:<28} {conjunto:<12} "
              f"{vals['RMSE']:>8.4f} {vals['MAE']:>8.4f}")

print(f"\n  ▸ CLASSIFICAÇÃO (Accuracy / F1-Score)")
print(f"  {'Modelo':<28} {'Conjunto':<12} {'Accuracy':>10} {'F1-Score':>10}")
print(f"  {SEP2[:62]}")
for nome_modelo, metricas in [
    ("Árvore de Decisão", metricas_dt_clf),
    ("Random Forest",     metricas_rf_clf),
]:
    for conjunto, vals in metricas.items():
        print(f"  {nome_modelo:<28} {conjunto:<12} "
              f"{vals['Accuracy']:>10.4f} {vals['F1-Score']:>10.4f}")

print(f"\n{SEP}")
print("  MELHORES HIPERPARÂMETROS ENCONTRADOS:")
print(f"  Árvore Regressão    : {grid_dt_reg.best_params_}")
print(f"  Árvore Classif.     : {grid_dt_clf.best_params_}")
print(f"  RF Regressão        : {grid_rf_reg.best_params_}")
print(f"  RF Classif.         : {grid_rf_clf.best_params_}")

# Guardar CSV com resultados para tabela comparativa do Sprint 3
linhas = []
for tarefa, pares in [
    ("Regressão",     [
        ("Árvore de Decisão", metricas_dt_reg),
        ("Random Forest",     metricas_rf_reg),
    ]),
    ("Classificação", [
        ("Árvore de Decisão", metricas_dt_clf),
        ("Random Forest",     metricas_rf_clf),
    ]),
]:
    for nome_modelo, metricas in pares:
        for conjunto, vals in metricas.items():
            linha = {"Tarefa": tarefa, "Modelo": nome_modelo, "Conjunto": conjunto}
            linha.update(vals)
            linhas.append(linha)

df_resultados = pd.DataFrame(linhas)
caminho_csv = os.path.join("resultados", "resultados_modelos_arvores_rf.csv")
df_resultados.to_csv(caminho_csv, index=False)
print(f"\n  [✔] Resultados guardados → {caminho_csv}")

print(f"\n{SEP}")
print("  MODELOS GUARDADOS EM: modelos/")
print("    • modelo_arvore_regressao.pickle")
print("    • modelo_arvore_classificacao.pickle")
print("    • modelo_rf_regressao.pickle")
print("    • modelo_rf_classificacao.pickle")
print("  GRÁFICOS GUARDADOS EM: resultados/graficos/")
print("    • rf_feature_importance_regressao.png")
print("    • rf_feature_importance_classificacao.png")
print("    • comparacao_feature_importance.png")
print("    • feature_importance_por_categoria.png  ← responde à questão de negócio")
print("    • arvore_decisao_visualizacao.png")
print(SEP)
print("\n  Treinamento Membro 2 — Sprint 3 concluído")
print(SEP + "\n")
