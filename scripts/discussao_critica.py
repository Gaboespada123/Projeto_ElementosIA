"""
=============================================================================
  DISCUSSÃO CRÍTICA DOS RESULTADOS
  Comparação unificada de todos os modelos do Sprint 3
=============================================================================
  Carrega os 8 modelos treinados, avalia-os no conjunto de teste,
  produz a tabela comparativa e os gráficos de análise crítica.

  Input  : resultados/dados_limpos_final.csv
           resultados/splits_info.pickle
           modelos/*.pickle  (8 modelos)
  Output : resultados/graficos/critica_rmse_comparacao.png
           resultados/graficos/critica_f1_comparacao.png
           resultados/graficos/critica_overfitting.png
           resultados/graficos/critica_predicao_vs_real.png
           resultados/graficos/critica_feature_selection.png
           resultados/tabela_comparativa_todos_modelos.csv
=============================================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")

try:
    from sklearn.metrics import root_mean_squared_error as _rmse_fn
    def rmse(y_true, y_pred):
        return float(_rmse_fn(y_true, y_pred))
except ImportError:
    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


os.makedirs(os.path.join("resultados", "graficos"), exist_ok=True)
GRAFICOS = os.path.join("resultados", "graficos")
SEP  = "=" * 70
SEP2 = "-" * 70

print(SEP)
print("  DISCUSSÃO CRÍTICA — Comparação de todos os modelos do Sprint 3")
print(SEP)


# =============================================================================
# 1. CARREGAR DADOS E RECRIAR SPLITS
# =============================================================================

df_raw = pd.read_csv(os.path.join("resultados", "dados_limpos_final.csv"))
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated(keep="first")]

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


def _extrair_indice(d, *chaves):
    for chave in chaves:
        if chave in d:
            return d[chave]
    raise KeyError(f"Nenhuma das chaves {chaves} encontrada. Chaves: {list(d.keys())}")


with open(os.path.join("resultados", "splits_info.pickle"), "rb") as f:
    splits_info = pickle.load(f)

idx_treino = _extrair_indice(splits_info, "indice_treino", "train_index")
idx_val    = _extrair_indice(splits_info, "indice_val",   "val_index")
idx_teste  = _extrair_indice(splits_info, "indice_teste", "test_index")

X_treino = X_total.iloc[idx_treino].reset_index(drop=True)
X_val    = X_total.iloc[idx_val   ].reset_index(drop=True)
X_teste  = X_total.iloc[idx_teste ].reset_index(drop=True)
y_treino = y_total.iloc[idx_treino].reset_index(drop=True)
y_val    = y_total.iloc[idx_val   ].reset_index(drop=True)
y_teste  = y_total.iloc[idx_teste ].reset_index(drop=True)

mediana_global = y_total.median()
y_bin_treino = (y_treino > mediana_global).astype(int)
y_bin_val    = (y_val    > mediana_global).astype(int)
y_bin_teste  = (y_teste  > mediana_global).astype(int)

print(f"\n  Dataset : {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas")
print(f"  Treino  : {len(X_treino)} | Validação: {len(X_val)} | Teste: {len(X_teste)}")


# =============================================================================
# 2. CARREGAR TODOS OS MODELOS
# =============================================================================

def _carregar(caminho):
    if not os.path.exists(caminho):
        print(f"  [!] Modelo não encontrado: {caminho} — será ignorado")
        return None
    with open(caminho, "rb") as f:
        return pickle.load(f)


modelos_reg = {
    "Reg. Linear"     : _carregar("modelos/modelo_regressao_linear.pickle"),
    "Árvore Decisão"  : _carregar("modelos/modelo_arvore_regressao.pickle"),
    "Random Forest"   : _carregar("modelos/modelo_rf_regressao.pickle"),
    "XGBoost"         : _carregar("modelos/modelo_xgb_regressao.pickle"),
}
modelos_clf = {
    "Reg. Logística"  : _carregar("modelos/modelo_regressao_logistica.pickle"),
    "Árvore Decisão"  : _carregar("modelos/modelo_arvore_classificacao.pickle"),
    "Random Forest"   : _carregar("modelos/modelo_rf_classificacao.pickle"),
    "XGBoost"         : _carregar("modelos/modelo_xgb_classificacao.pickle"),
}

modelos_reg = {k: v for k, v in modelos_reg.items() if v is not None}
modelos_clf = {k: v for k, v in modelos_clf.items() if v is not None}

print(f"\n  Modelos de regressão   : {list(modelos_reg.keys())}")
print(f"  Modelos de classificação: {list(modelos_clf.keys())}")


# =============================================================================
# 3. CALCULAR MÉTRICAS EM TREINO, VALIDAÇÃO E TESTE
# =============================================================================

res_reg = {}
for nome, modelo in modelos_reg.items():
    res_reg[nome] = {}
    for conj, X_s, y_s in [
        ("Treino",    X_treino, y_treino),
        ("Validação", X_val,    y_val),
        ("Teste",     X_teste,  y_teste),
    ]:
        pred = modelo.predict(X_s)
        res_reg[nome][conj] = {
            "RMSE": round(rmse(y_s, pred), 4),
            "MAE" : round(float(mean_absolute_error(y_s, pred)), 4),
        }

res_clf = {}
for nome, modelo in modelos_clf.items():
    res_clf[nome] = {}
    for conj, X_s, y_s in [
        ("Treino",    X_treino, y_bin_treino),
        ("Validação", X_val,    y_bin_val),
        ("Teste",     X_teste,  y_bin_teste),
    ]:
        pred = modelo.predict(X_s)
        res_clf[nome][conj] = {
            "Accuracy": round(float(accuracy_score(y_s, pred)), 4),
            "F1-Score": round(float(f1_score(y_s, pred, zero_division=0)), 4),
        }


# =============================================================================
# 4. TABELA COMPARATIVA IMPRESSA
# =============================================================================

print(f"\n\n{SEP}")
print("  REGRESSÃO — RMSE (menor = melhor)")
print(SEP)
print(f"  {'Modelo':<20} {'RMSE Treino':>12} {'RMSE Val':>10} {'RMSE Teste':>11} {'MAE Teste':>10}")
print(f"  {SEP2[:64]}")
for nome, res in res_reg.items():
    over = "⚠" if res["Teste"]["RMSE"] / max(res["Treino"]["RMSE"], 1e-9) > 2.5 else " "
    print(f"  {nome:<20} {res['Treino']['RMSE']:>12.4f} "
          f"{res['Validação']['RMSE']:>10.4f} "
          f"{res['Teste']['RMSE']:>11.4f} "
          f"{res['Teste']['MAE']:>10.4f}  {over}")

print(f"\n{SEP}")
print("  CLASSIFICAÇÃO — F1-Score e Accuracy (maior = melhor)")
print(SEP)
print(f"  {'Modelo':<20} {'Acc Treino':>11} {'Acc Val':>9} {'Acc Teste':>10} {'F1 Teste':>10}")
print(f"  {SEP2[:62]}")
for nome, res in res_clf.items():
    over = "⚠" if res["Treino"]["F1-Score"] - res["Teste"]["F1-Score"] > 0.15 else " "
    print(f"  {nome:<20} {res['Treino']['Accuracy']:>11.4f} "
          f"{res['Validação']['Accuracy']:>9.4f} "
          f"{res['Teste']['Accuracy']:>10.4f} "
          f"{res['Teste']['F1-Score']:>10.4f}  {over}")
print("  ⚠ = possível overfitting (gap treino-teste elevado)")


# =============================================================================
# 5. GUARDAR CSV UNIFICADO
# =============================================================================

linhas = []
for tarefa, resultados in [("Regressão", res_reg), ("Classificação", res_clf)]:
    for nome, res in resultados.items():
        for conj, vals in res.items():
            linha = {"Tarefa": tarefa, "Modelo": nome, "Conjunto": conj}
            linha.update(vals)
            linhas.append(linha)

df_comp = pd.DataFrame(linhas)
caminho_csv = os.path.join("resultados", "tabela_comparativa_todos_modelos.csv")
df_comp.to_csv(caminho_csv, index=False)
print(f"\n  [✔] Tabela guardada → {caminho_csv}")


# =============================================================================
# 6. GRÁFICOS
# =============================================================================

# Paleta: verde = interpretável (preferido enunciado), azul = ensemble, roxo = boosting, vermelho = linear
COR = {
    "Reg. Linear"    : "#e74c3c",
    "Reg. Logística" : "#e74c3c",
    "Árvore Decisão" : "#27ae60",
    "Random Forest"  : "#2980b9",
    "XGBoost"        : "#8e44ad",
}

legenda_patches = [
    mpatches.Patch(color="#27ae60", label="Interpretável — preferido pelo enunciado"),
    mpatches.Patch(color="#2980b9", label="Ensemble (Random Forest)"),
    mpatches.Patch(color="#8e44ad", label="Boosting (XGBoost)"),
    mpatches.Patch(color="#e74c3c", label="Modelo linear (baseline)"),
]


def _spine_off(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─── Gráfico 1: RMSE no Teste — Regressão ────────────────────────────────────
nomes_r = list(res_reg.keys())
vals_r  = [res_reg[n]["Teste"]["RMSE"] for n in nomes_r]
cores_r = [COR.get(n, "#95a5a6") for n in nomes_r]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(nomes_r, vals_r, color=cores_r, edgecolor="white", linewidth=1.2, width=0.55)
for bar, val in zip(bars, vals_r):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("RMSE no Teste (kg)", fontsize=11)
ax.set_title("Comparação de RMSE — Regressão\n(menor é melhor | verde = modelo interpretável preferido)",
             fontsize=12, fontweight="bold")
ax.legend(handles=legenda_patches, fontsize=8, loc="upper right")
_spine_off(ax)
plt.tight_layout()
c = os.path.join(GRAFICOS, "critica_rmse_comparacao.png")
plt.savefig(c, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  [✔] Gráfico → {c}")


# ─── Gráfico 2: F1-Score no Teste — Classificação ────────────────────────────
nomes_c = list(res_clf.keys())
vals_c  = [res_clf[n]["Teste"]["F1-Score"] for n in nomes_c]
cores_c = [COR.get(n, "#95a5a6") for n in nomes_c]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(nomes_c, vals_c, color=cores_c, edgecolor="white", linewidth=1.2, width=0.55)
for bar, val in zip(bars, vals_c):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("F1-Score no Teste", fontsize=11)
ax.set_ylim(0, min(1.05, max(vals_c) * 1.2))
ax.set_title("Comparação de F1-Score — Classificação\n(maior é melhor | verde = modelo interpretável preferido)",
             fontsize=12, fontweight="bold")
ax.legend(handles=legenda_patches, fontsize=8, loc="lower right")
_spine_off(ax)
plt.tight_layout()
c = os.path.join(GRAFICOS, "critica_f1_comparacao.png")
plt.savefig(c, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [✔] Gráfico → {c}")


# ─── Gráfico 3: Overfitting — Treino vs Validação vs Teste ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
w = 0.28

for ax, (titulo, resultados, metrica, label_y) in zip(
    axes,
    [
        ("Regressão — RMSE",      res_reg, "RMSE",     "RMSE (kg)"),
        ("Classificação — F1",    res_clf, "F1-Score", "F1-Score"),
    ],
):
    nomes = list(resultados.keys())
    v_tr  = [resultados[n]["Treino"]["RMSE" if metrica == "RMSE" else "F1-Score"]    for n in nomes]
    v_val = [resultados[n]["Validação"]["RMSE" if metrica == "RMSE" else "F1-Score"] for n in nomes]
    v_te  = [resultados[n]["Teste"]["RMSE" if metrica == "RMSE" else "F1-Score"]     for n in nomes]
    x = np.arange(len(nomes))
    ax.bar(x - w, v_tr,  w, label="Treino",    color="#2ecc71", alpha=0.85)
    ax.bar(x,     v_val, w, label="Validação", color="#f39c12", alpha=0.85)
    ax.bar(x + w, v_te,  w, label="Teste",     color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel(label_y, fontsize=10)
    ax.set_title(f"{titulo}\n(gap Treino→Teste grande = overfitting)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    _spine_off(ax)

fig.suptitle("Análise de Overfitting — Todos os Modelos", fontsize=13, fontweight="bold")
plt.tight_layout()
c = os.path.join(GRAFICOS, "critica_overfitting.png")
plt.savefig(c, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [✔] Gráfico → {c}")


# ─── Gráfico 4: Previsão vs Real (Árvore de Decisão — modelo preferido) ──────
nome_dt = "Árvore Decisão"
if nome_dt in modelos_reg:
    modelo_dt = modelos_reg[nome_dt]
    y_pred = modelo_dt.predict(X_teste)
    rmse_dt  = res_reg[nome_dt]["Teste"]["RMSE"]
    mae_dt   = res_reg[nome_dt]["Teste"]["MAE"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_teste, y_pred, alpha=0.45, color="#27ae60",
               edgecolors="white", linewidth=0.5, s=45)
    lim = [min(float(y_teste.min()), float(y_pred.min())) - 0.2,
           max(float(y_teste.max()), float(y_pred.max())) + 0.2]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Previsão perfeita")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Valor Real (kg)", fontsize=11)
    ax.set_ylabel("Valor Previsto (kg)", fontsize=11)
    ax.set_title(
        f"Árvore de Decisão — Previsto vs Real (Conjunto de Teste)\n"
        f"RMSE: {rmse_dt:.4f} kg  |  MAE: {mae_dt:.4f} kg",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    _spine_off(ax)
    plt.tight_layout()
    c = os.path.join(GRAFICOS, "critica_predicao_vs_real.png")
    plt.savefig(c, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✔] Gráfico → {c}")


# ─── Gráfico 5: Feature Selection — valem todas as variáveis? ────────────────
# Usa importâncias do Random Forest para selecionar top N features
# Treina Árvore de Decisão com diferentes subconjuntos e compara RMSE
if "Random Forest" in modelos_reg:
    rf = modelos_reg["Random Forest"]
    fi = pd.Series(rf.feature_importances_, index=X_total.columns).sort_values(ascending=False)

    top_ns   = [5, 10, 15, 20, 30, len(X_total.columns)]
    rmse_sel = []

    for n in top_ns:
        feats = fi.head(n).index.tolist()
        dt_sel = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
        dt_sel.fit(X_treino[feats], y_treino)
        pred_sel = dt_sel.predict(X_teste[feats])
        rmse_sel.append(round(rmse(y_teste, pred_sel), 4))

    labels = [f"Top {n}" if n < len(X_total.columns) else f"Todas\n({n})" for n in top_ns]
    cores_sel = ["#e74c3c", "#f39c12", "#2ecc71", "#27ae60", "#2980b9", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, rmse_sel, color=cores_sel, edgecolor="white", linewidth=1.2, width=0.6)
    for bar, val in zip(bars, rmse_sel):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("RMSE no Teste (kg)", fontsize=11)
    ax.set_xlabel("Nº de variáveis usadas (as mais importantes pelo RF)", fontsize=10)
    ax.set_title(
        "Feature Selection — Vale usar todas as variáveis?\n"
        "(Árvore de Decisão treinada com diferentes subconjuntos)",
        fontsize=11, fontweight="bold",
    )
    _spine_off(ax)
    plt.tight_layout()
    c = os.path.join(GRAFICOS, "critica_feature_selection.png")
    plt.savefig(c, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✔] Gráfico → {c}")

    print(f"\n  Feature Selection (RMSE Teste por nº de variáveis):")
    for n, r in zip(top_ns, rmse_sel):
        label = f"Top {n:>2}" if n < len(X_total.columns) else f"Todas ({n})"
        diff  = r - min(rmse_sel)
        print(f"    {label}: RMSE = {r:.4f}  (+{diff:.4f} vs melhor)")


# =============================================================================
# 7. DISCUSSÃO CRÍTICA AUTOMÁTICA
# =============================================================================

print(f"\n\n{SEP}")
print("  DISCUSSÃO CRÍTICA DOS RESULTADOS")
print(SEP)

melhor_reg  = min(res_reg, key=lambda n: res_reg[n]["Teste"]["RMSE"])
melhor_clf  = max(res_clf, key=lambda n: res_clf[n]["Teste"]["F1-Score"])
pior_reg    = max(res_reg, key=lambda n: res_reg[n]["Teste"]["RMSE"])

rmse_tr_best = res_reg[melhor_reg]["Treino"]["RMSE"]
rmse_te_best = res_reg[melhor_reg]["Teste"]["RMSE"]
f1_tr_best   = res_clf[melhor_clf]["Treino"]["F1-Score"]
f1_te_best   = res_clf[melhor_clf]["Teste"]["F1-Score"]

print(f"\n  ▸ MELHOR MODELO — REGRESSÃO  : {melhor_reg}")
print(f"    RMSE Treino = {rmse_tr_best:.4f}  |  RMSE Teste = {rmse_te_best:.4f}")
ratio_r = rmse_te_best / max(rmse_tr_best, 1e-9)
flag_r  = "⚠ Overfitting provável" if ratio_r > 2.5 else "✔ Generalização aceitável"
print(f"    Rácio Teste/Treino = {ratio_r:.2f}x  →  {flag_r}")

print(f"\n  ▸ MELHOR MODELO — CLASSIFICAÇÃO: {melhor_clf}")
print(f"    F1 Treino = {f1_tr_best:.4f}  |  F1 Teste = {f1_te_best:.4f}")
gap_c = f1_tr_best - f1_te_best
flag_c = "⚠ Overfitting provável" if gap_c > 0.15 else "✔ Generalização aceitável"
print(f"    Gap F1 Treino-Teste = {gap_c:.4f}  →  {flag_c}")

print(f"\n  ▸ MODELO PREFERIDO PELO ENUNCIADO (interpretável): Árvore de Decisão")
if "Árvore Decisão" in res_reg:
    rmse_dt = res_reg["Árvore Decisão"]["Teste"]["RMSE"]
    rmse_best = res_reg[melhor_reg]["Teste"]["RMSE"]
    dif_pct = (rmse_dt - rmse_best) / max(rmse_best, 1e-9) * 100
    if dif_pct <= 5:
        print(f"    A Árvore de Decisão tem RMSE = {rmse_dt:.4f}, praticamente igual ao melhor "
              f"modelo ({melhor_reg}, RMSE = {rmse_best:.4f}).")
        print("    → Conclusão: usar a Árvore de Decisão é justificado — é interpretável e "
              "tem performance comparável.")
    else:
        print(f"    A Árvore de Decisão tem RMSE = {rmse_dt:.4f} vs {rmse_best:.4f} do melhor "
              f"modelo ({melhor_reg}) — diferença de {dif_pct:.1f}%.")
        print("    → Conclusão: há tradeoff entre interpretabilidade e performance.")

print(f"\n  ▸ PIOR MODELO (baseline): {pior_reg}")
print(f"    RMSE Teste = {res_reg[pior_reg]['Teste']['RMSE']:.4f} — serve de referência mínima.")

print(f"\n{SEP}")
print("  MODELOS GUARDADOS  : modelos/")
print("  GRÁFICOS GUARDADOS : resultados/graficos/")
print("    • critica_rmse_comparacao.png")
print("    • critica_f1_comparacao.png")
print("    • critica_overfitting.png")
print("    • critica_predicao_vs_real.png")
print("    • critica_feature_selection.png")
print("  CSV                : resultados/tabela_comparativa_todos_modelos.csv")
print(SEP)
print("\n  Discussão Crítica concluída.")
print(SEP + "\n")
