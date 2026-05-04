"""
=============================================================================
  PREVISÃO INTERATIVA — Respostas às perguntas do enunciado
=============================================================================
  Usa a Árvore de Decisão (modelo interpretável preferido pelo enunciado)
  para responder diretamente às questões práticas do trabalho:

  Q1 — "Quanto peso vou perder se fizer dieta Keto (low_carb)?"
       → Compara a variação de peso prevista para cada tipo de dieta

  Q2 — "Qual o tipo de nutricionista que maximiza a minha perda de peso?"
       → Compara o efeito de cada abordagem e especialidade do nutricionista

  Q3 — Perfil de paciente ideal: que combinação dieta × nutricionista
       produz a maior perda de peso esperada?

  Input  : resultados/dados_limpos_final.csv
           resultados/dataset_limpo.csv        (valores originais)
           resultados/splits_info.pickle
           modelos/modelo_arvore_regressao.pickle
           modelos/modelo_rf_regressao.pickle   (confirmação)
  Output : resultados/graficos/previsao_por_tipo_dieta.png
           resultados/graficos/previsao_por_abordagem_nutricionista.png
           resultados/graficos/previsao_por_especialidade_nutricionista.png
           resultados/graficos/previsao_melhor_combinacao.png
=============================================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

os.makedirs(os.path.join("resultados", "graficos"), exist_ok=True)
GRAFICOS = os.path.join("resultados", "graficos")
SEP  = "=" * 68
SEP2 = "-" * 68

print(SEP)
print("  PREVISÃO INTERATIVA — Respostas às perguntas do enunciado")
print(SEP)


# =============================================================================
# 1. CARREGAR DATASETS
#    dataset_limpo.csv   → valores originais (diet_type, approach, specialty)
#    dados_limpos_final.csv → features preprocessadas para os modelos
# =============================================================================

df_limpo = pd.read_csv(os.path.join("resultados", "dataset_limpo.csv"))
df_final = pd.read_csv(os.path.join("resultados", "dados_limpos_final.csv"))

# Normalizar dataset_limpo (valores categóricos)
for col in ["diet_type", "approach", "specialty", "sex"]:
    if col in df_limpo.columns:
        df_limpo[col] = df_limpo[col].astype(str).str.strip().str.lower()

# Normalizar colunas do dataset final (mesma lógica dos outros scripts)
df_final.columns = df_final.columns.str.strip()
df_final = df_final.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
df_final = df_final.loc[:, ~df_final.columns.duplicated(keep="first")]
for col in df_final.columns:
    if df_final[col].dtype == object:
        unicos = df_final[col].dropna().str.strip().str.lower().unique()
        if set(unicos).issubset({"true", "false"}):
            df_final[col] = df_final[col].str.strip().str.lower().map({"true": 1, "false": 0})
df_final = df_final.apply(pd.to_numeric, errors="coerce")
df_final = df_final.fillna(df_final.median(numeric_only=True))

COLUNA_ALVO = "weight_change_kg_6m"
X_total = df_final.drop(columns=[COLUNA_ALVO])

print(f"\n  Dataset processado : {df_final.shape[0]} linhas × {df_final.shape[1]} colunas")
print(f"  Dataset original   : {df_limpo.shape[0]} linhas × {df_limpo.shape[1]} colunas")


# =============================================================================
# 2. CARREGAR SPLITS (índices do conjunto de teste)
# =============================================================================

def _extrair_indice(d, *chaves):
    for chave in chaves:
        if chave in d:
            return d[chave]
    raise KeyError(f"Chaves {chaves} não encontradas. Disponíveis: {list(d.keys())}")


with open(os.path.join("resultados", "splits_info.pickle"), "rb") as f:
    splits_info = pickle.load(f)

idx_teste = _extrair_indice(splits_info, "indice_teste", "test_index")

# Conjunto de teste: features preprocessadas + valores categóricos originais
X_teste   = X_total.iloc[idx_teste].reset_index(drop=True)
cat_teste  = df_limpo.iloc[idx_teste].reset_index(drop=True)

print(f"  Conjunto de teste  : {len(X_teste)} amostras")


# =============================================================================
# 3. CARREGAR MODELOS
#    Prioritiza Árvore de Decisão (preferida pelo enunciado por ser interpretável)
#    Usa também o Random Forest como confirmação
# =============================================================================

def _carregar(caminho):
    if not os.path.exists(caminho):
        print(f"  [!] Não encontrado: {caminho}")
        return None
    with open(caminho, "rb") as f:
        return pickle.load(f)


modelo_dt = _carregar("modelos/modelo_arvore_regressao.pickle")
modelo_rf = _carregar("modelos/modelo_rf_regressao.pickle")

if modelo_dt is None and modelo_rf is None:
    print("  Erro: nenhum modelo de regressão encontrado. Corre primeiro o Sprint 3.")
    raise SystemExit(1)

modelo_principal = modelo_dt if modelo_dt is not None else modelo_rf
nome_principal   = "Árvore de Decisão" if modelo_dt is not None else "Random Forest"
print(f"\n  Modelo principal: {nome_principal} (interpretável)")


# =============================================================================
# 4. FUNÇÃO AUXILIAR
#    Para cada grupo (diet_type, approach, specialty), filtra os pacientes
#    do conjunto de teste com essa característica e calcula:
#      - Variação de peso REAL (média)
#      - Variação de peso PREVISTA pelo modelo (média)
# =============================================================================

def comparar_grupos(coluna_original, grupos, X_test, cat_test, modelo):
    """
    Retorna DataFrame com média real e prevista por grupo.
    grupos: lista dos valores únicos da coluna a comparar.
    """
    y_real_total = df_final[COLUNA_ALVO].iloc[idx_teste].reset_index(drop=True)

    linhas = []
    for grupo in grupos:
        mask = cat_test[coluna_original] == grupo
        if mask.sum() == 0:
            continue
        X_grp  = X_test[mask]
        y_real = y_real_total[mask]
        y_prev = modelo.predict(X_grp)
        linhas.append({
            "Grupo"         : grupo,
            "N"             : int(mask.sum()),
            "Real (média)"  : round(float(y_real.mean()), 4),
            "Previsto (média)": round(float(y_prev.mean()), 4),
            "Diff (Prev-Real)": round(float(y_prev.mean()) - float(y_real.mean()), 4),
        })
    return pd.DataFrame(linhas).sort_values("Previsto (média)")


def _spine_off(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# 5. Q1 — TIPO DE DIETA
#    "Quanto peso vou perder se fizer dieta Keto (low_carb)?"
# =============================================================================

print(f"\n{SEP2}")
print("  Q1 — Comparação por tipo de dieta")
print(SEP2)

tipos_dieta = sorted(cat_teste["diet_type"].dropna().unique().tolist())
df_dieta = comparar_grupos("diet_type", tipos_dieta, X_teste, cat_teste, modelo_principal)

print(f"\n  {'Tipo de Dieta':<20} {'N':>5} {'Real (kg)':>11} {'Previsto (kg)':>14}")
print(f"  {SEP2[:52]}")
for _, row in df_dieta.sort_values("Previsto (média)").iterrows():
    melhor = " ← menor variação (mais perda)" if row["Previsto (média)"] == df_dieta["Previsto (média)"].min() else ""
    print(f"  {row['Grupo']:<20} {row['N']:>5} {row['Real (média)']:>11.4f} "
          f"{row['Previsto (média)']:>14.4f}{melhor}")

# Gráfico Q1
fig, ax = plt.subplots(figsize=(9, 5))
grupos_d = df_dieta["Grupo"].tolist()
real_d   = df_dieta["Real (média)"].tolist()
prev_d   = df_dieta["Previsto (média)"].tolist()
x = np.arange(len(grupos_d))
w = 0.38
ax.bar(x - w / 2, real_d,   w, label="Valor Real (média teste)",    color="#e74c3c", alpha=0.85)
ax.bar(x + w / 2, prev_d,   w, label=f"Previsto pelo {nome_principal}", color="#27ae60", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([g.replace("_", " ").title() for g in grupos_d], fontsize=10)
ax.set_ylabel("Variação de Peso (kg)", fontsize=11)
ax.set_title(
    "Q1 — Quanto peso se perde por tipo de dieta?\n"
    "(valores negativos = perda de peso; mais negativo = mais eficaz)",
    fontsize=12, fontweight="bold",
)
ax.legend(fontsize=9)
_spine_off(ax)

# Anotação: melhor dieta
melhor_dieta = df_dieta.loc[df_dieta["Previsto (média)"].idxmin(), "Grupo"]
print(f"\n  Conclusão Q1: A dieta '{melhor_dieta.replace('_',' ').title()}' está associada "
      f"à maior perda de peso prevista pelo modelo.")

plt.tight_layout()
c = os.path.join(GRAFICOS, "previsao_por_tipo_dieta.png")
plt.savefig(c, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [✔] Gráfico → {c}")


# =============================================================================
# 6. Q2 — ABORDAGEM DO NUTRICIONISTA
#    "Qual o tipo de nutricionista que maximiza a minha perda de peso?"
# =============================================================================

print(f"\n{SEP2}")
print("  Q2 — Comparação por abordagem do nutricionista")
print(SEP2)

abordagens = sorted(cat_teste["approach"].dropna().unique().tolist())
df_abord = comparar_grupos("approach", abordagens, X_teste, cat_teste, modelo_principal)

print(f"\n  {'Abordagem':<22} {'N':>5} {'Real (kg)':>11} {'Previsto (kg)':>14}")
print(f"  {SEP2[:54]}")
for _, row in df_abord.sort_values("Previsto (média)").iterrows():
    melhor = " ← melhor resultado" if row["Previsto (média)"] == df_abord["Previsto (média)"].min() else ""
    print(f"  {row['Grupo']:<22} {row['N']:>5} {row['Real (média)']:>11.4f} "
          f"{row['Previsto (média)']:>14.4f}{melhor}")

# Gráfico Q2
fig, ax = plt.subplots(figsize=(10, 5))
grupos_a = df_abord["Grupo"].tolist()
real_a   = df_abord["Real (média)"].tolist()
prev_a   = df_abord["Previsto (média)"].tolist()
x = np.arange(len(grupos_a))
ax.bar(x - w / 2, real_a, w, label="Valor Real (média teste)",    color="#e74c3c", alpha=0.85)
ax.bar(x + w / 2, prev_a, w, label=f"Previsto pelo {nome_principal}", color="#2980b9", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([g.replace("_", " ").title() for g in grupos_a], fontsize=10)
ax.set_ylabel("Variação de Peso (kg)", fontsize=11)
ax.set_title(
    "Q2 — Qual abordagem do nutricionista maximiza a perda de peso?\n"
    "(valores negativos = perda de peso; mais negativo = mais eficaz)",
    fontsize=12, fontweight="bold",
)
ax.legend(fontsize=9)
_spine_off(ax)

melhor_abordagem = df_abord.loc[df_abord["Previsto (média)"].idxmin(), "Grupo"]
print(f"\n  Conclusão Q2: A abordagem '{melhor_abordagem.replace('_',' ').title()}' está associada "
      f"à maior perda de peso prevista.")

plt.tight_layout()
c = os.path.join(GRAFICOS, "previsao_por_abordagem_nutricionista.png")
plt.savefig(c, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [✔] Gráfico → {c}")


# =============================================================================
# 7. Q2b — ESPECIALIDADE DO NUTRICIONISTA
# =============================================================================

print(f"\n{SEP2}")
print("  Q2b — Comparação por especialidade do nutricionista")
print(SEP2)

especialidades = sorted(cat_teste["specialty"].dropna().unique().tolist())
df_espec = comparar_grupos("specialty", especialidades, X_teste, cat_teste, modelo_principal)

print(f"\n  {'Especialidade':<22} {'N':>5} {'Real (kg)':>11} {'Previsto (kg)':>14}")
print(f"  {SEP2[:54]}")
for _, row in df_espec.sort_values("Previsto (média)").iterrows():
    melhor = " ← melhor resultado" if row["Previsto (média)"] == df_espec["Previsto (média)"].min() else ""
    print(f"  {row['Grupo']:<22} {row['N']:>5} {row['Real (média)']:>11.4f} "
          f"{row['Previsto (média)']:>14.4f}{melhor}")

# Gráfico Q2b
fig, ax = plt.subplots(figsize=(10, 5))
grupos_e = df_espec["Grupo"].tolist()
real_e   = df_espec["Real (média)"].tolist()
prev_e   = df_espec["Previsto (média)"].tolist()
x = np.arange(len(grupos_e))
ax.bar(x - w / 2, real_e, w, label="Valor Real (média teste)",    color="#e74c3c", alpha=0.85)
ax.bar(x + w / 2, prev_e, w, label=f"Previsto pelo {nome_principal}", color="#8e44ad", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([g.replace("_", " ").title() for g in grupos_e], fontsize=10)
ax.set_ylabel("Variação de Peso (kg)", fontsize=11)
ax.set_title(
    "Q2b — Qual especialidade do nutricionista maximiza a perda de peso?\n"
    "(valores negativos = perda de peso; mais negativo = mais eficaz)",
    fontsize=12, fontweight="bold",
)
ax.legend(fontsize=9)
_spine_off(ax)

melhor_espec = df_espec.loc[df_espec["Previsto (média)"].idxmin(), "Grupo"]
print(f"\n  Conclusão Q2b: A especialidade '{melhor_espec.replace('_',' ').title()}' está "
      f"associada à maior perda de peso prevista.")

plt.tight_layout()
c = os.path.join(GRAFICOS, "previsao_por_especialidade_nutricionista.png")
plt.savefig(c, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [✔] Gráfico → {c}")


# =============================================================================
# 8. Q3 — MELHOR COMBINAÇÃO DIETA × ABORDAGEM
#    "Que combinação maximiza a perda de peso?"
# =============================================================================

print(f"\n{SEP2}")
print("  Q3 — Melhor combinação: tipo de dieta × abordagem do nutricionista")
print(SEP2)

y_real_total = df_final[COLUNA_ALVO].iloc[idx_teste].reset_index(drop=True)

combinacoes = []
for dieta in tipos_dieta:
    for abord in abordagens:
        mask = (cat_teste["diet_type"] == dieta) & (cat_teste["approach"] == abord)
        if mask.sum() < 5:
            continue
        X_grp  = X_teste[mask]
        y_prev = modelo_principal.predict(X_grp)
        y_real = y_real_total[mask]
        combinacoes.append({
            "Dieta"           : dieta.replace("_", " ").title(),
            "Abordagem"       : abord.replace("_", " ").title(),
            "N"               : int(mask.sum()),
            "Previsto (média)": round(float(y_prev.mean()), 4),
            "Real (média)"    : round(float(y_real.mean()), 4),
        })

df_comb = pd.DataFrame(combinacoes).sort_values("Previsto (média)")

print(f"\n  Top 5 melhores combinações (maior perda de peso prevista):")
print(f"  {'Dieta':<22} {'Abordagem':<20} {'N':>4} {'Previsto (kg)':>14} {'Real (kg)':>10}")
print(f"  {SEP2[:72]}")
for _, row in df_comb.head(5).iterrows():
    print(f"  {row['Dieta']:<22} {row['Abordagem']:<20} {row['N']:>4} "
          f"{row['Previsto (média)']:>14.4f} {row['Real (média)']:>10.4f}")

# Gráfico Q3 — Heatmap de combinações
if len(df_comb) > 0:
    pivot = df_comb.pivot_table(
        index="Abordagem", columns="Dieta",
        values="Previsto (média)", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticklabels(pivot.index, fontsize=10)
    plt.colorbar(im, ax=ax, label="Variação de peso prevista (kg)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black", fontweight="bold")
    ax.set_xlabel("Tipo de Dieta", fontsize=11)
    ax.set_ylabel("Abordagem do Nutricionista", fontsize=11)
    ax.set_title(
        "Q3 — Melhor combinação: Dieta × Abordagem do Nutricionista\n"
        "(verde = mais perda de peso | vermelho = menos eficaz)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    c = os.path.join(GRAFICOS, "previsao_melhor_combinacao.png")
    plt.savefig(c, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [✔] Gráfico → {c}")

    melhor_row = df_comb.iloc[0]
    print(f"\n  Conclusão Q3: A melhor combinação é "
          f"'{melhor_row['Dieta']}' com nutricionista '{melhor_row['Abordagem']}'")
    print(f"  Variação de peso prevista: {melhor_row['Previsto (média)']:.4f} kg")


# =============================================================================
# 9. RESUMO FINAL
# =============================================================================

print(f"\n\n{SEP}")
print("  RESUMO DAS RESPOSTAS — PERGUNTAS DO ENUNCIADO")
print(SEP)
print(f"\n  Modelo usado: {nome_principal} (interpretável, preferido pelo enunciado)")
print(f"\n  Q1 — Melhor dieta para perda de peso:")
print(f"       → {melhor_dieta.replace('_',' ').title()}")
print(f"\n  Q2 — Melhor abordagem do nutricionista:")
print(f"       → {melhor_abordagem.replace('_',' ').title()}")
print(f"\n  Q2b— Melhor especialidade do nutricionista:")
print(f"       → {melhor_espec.replace('_',' ').title()}")
if len(df_comb) > 0:
    print(f"\n  Q3 — Melhor combinação dieta × nutricionista:")
    print(f"       → {melhor_row['Dieta']} + Nutricionista {melhor_row['Abordagem']}")
    print(f"         (Variação de peso prevista: {melhor_row['Previsto (média)']:.4f} kg)")
print(f"\n  GRÁFICOS GUARDADOS EM: resultados/graficos/")
print("    • previsao_por_tipo_dieta.png")
print("    • previsao_por_abordagem_nutricionista.png")
print("    • previsao_por_especialidade_nutricionista.png")
print("    • previsao_melhor_combinacao.png")
print(SEP)
print("\n  Previsão Interativa concluída.")
print(SEP + "\n")
