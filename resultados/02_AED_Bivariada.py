"""
=============================================================
  SPRINT 2 - AED BIVARIADA
  Membro 2: Cruzamentos entre pares de variáveis
=============================================================
  Input  : resultados/dataset_limpo.csv
  Output : resultados/graficos/bivariada_*.png
  Executar da raiz do projeto: python resultados/02_AED_Bivariada.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Configurações visuais ──────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── Caminhos e constantes ──────────────────────────────────────
INPUT_PATH   = "resultados/dataset_limpo.csv"
GRAFICOS_DIR = "resultados/graficos"
TARGET       = "weight_change_kg_6m"
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# ── 1. Carregar e enriquecer dados ─────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")

# Extrair estação do ano de record_created_at
if "record_created_at" in df.columns:
    df["mes"] = pd.to_datetime(df["record_created_at"], errors="coerce").dt.month
    def mes_para_estacao(m):
        if m in [12, 1, 2]:  return "inverno"
        elif m in [3, 4, 5]: return "primavera"
        elif m in [6, 7, 8]: return "verao"
        else:                 return "outono"
    df["estacao"] = df["mes"].map(mes_para_estacao)

# Variável binária de sucesso
if TARGET in df.columns:
    df["sucesso"] = (df[TARGET] < 0).astype(int)

# Remover IDs e data bruta
COLS_EXCLUIR = ["program_id", "patient_id", "nutritionist_id", "diet_id", "record_created_at"]
df = df.drop(columns=[c for c in COLS_EXCLUIR if c in df.columns])

# Normalizar categóricas
categ_cols = df.select_dtypes(include="object").columns.tolist()
for col in categ_cols:
    df[col] = df[col].str.strip().str.lower()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
print(f"Colunas numéricas: {len(numeric_cols)}  |  Categóricas: {categ_cols}\n")

# ── 2. Heatmap de correlações entre variáveis numéricas ────────
print("[1/7] A gerar heatmap de correlações...")
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    ax=ax, linewidths=0.5, annot_kws={"size": 8},
    cbar_kws={"label": "Coeficiente de Pearson"},
)
ax.set_title("Heatmap de Correlações — Variáveis Numéricas", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{GRAFICOS_DIR}/bivariada_heatmap_correlacoes.png", bbox_inches="tight")
plt.close()
print(f"   Guardado: {GRAFICOS_DIR}/bivariada_heatmap_correlacoes.png")

# ── 3. Ranking de correlação com a variável-alvo ──────────────
print("[2/7] A gerar ranking de correlações com o target...")
other_numeric = [c for c in numeric_cols if c not in [TARGET, "sucesso", "mes"]]
corr_target = df[other_numeric].corrwith(df[TARGET]).sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(10, max(6, len(other_numeric) * 0.45)))
colors = ["steelblue" if v >= 0 else "salmon" for v in corr_target.values]
bars = ax.barh(corr_target.index, corr_target.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title(f"Correlação de cada variável com '{TARGET}'", fontsize=13, fontweight="bold")
ax.set_xlabel("Coeficiente de Correlação de Pearson")
for bar, val in zip(bars, corr_target.values):
    offset = 0.005 if val >= 0 else -0.005
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", ha="left" if val >= 0 else "right", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{GRAFICOS_DIR}/bivariada_correlacao_target.png", bbox_inches="tight")
plt.close()
print(f"   Guardado: {GRAFICOS_DIR}/bivariada_correlacao_target.png")
print("\n  Top correlações com a variável-alvo:")
print(corr_target.head(8).to_string())

# ── 2b. Testes Estatísticos Formais ―――――――――――――――――――――――――――――――――
# Os gráficos mostram padrões visuais; os testes confirmam-nos com p-values.
# Regra: p < 0.05 indica diferença estatísticamente significativa.
print("\n" + "═" * 55)
print("  TESTES ESTATÍSTICOS")
print("═" * 55)

# Teste 1: Diferença de perda de peso entre sexos (Mann-Whitney U)
# Hipótese nula: distribuição de weight_change igual para homens e mulheres
if "sex" in df.columns and TARGET in df.columns:
    grupos_sexo = {s: df[df["sex"] == s][TARGET].dropna() for s in df["sex"].dropna().unique()}
    if len(grupos_sexo) == 2:
        keys = list(grupos_sexo.keys())
        stat, pval = stats.mannwhitneyu(grupos_sexo[keys[0]], grupos_sexo[keys[1]], alternative="two-sided")
        sig = "** Significativo" if pval < 0.05 else "Não significativo"
        print(f"\n  1. Mann-Whitney U: {keys[0]} vs {keys[1]} ({TARGET})")
        print(f"     U={stat:.1f},  p={pval:.4f}  →  {sig} (alpha=0.05)")
        for k, g in grupos_sexo.items():
            print(f"     {k}: média={g.mean():.3f} kg, mediana={g.median():.3f} kg, n={len(g)}")

# Teste 2: Diferença de resultado entre tipos de dieta (Kruskal-Wallis)
# Hipótese nula: distribuição de weight_change igual em todos os tipos de dieta
if "diet_type" in df.columns and TARGET in df.columns:
    grupos_dieta = [g[TARGET].dropna().values
                    for _, g in df.groupby("diet_type")
                    if len(g) >= 5]
    if len(grupos_dieta) >= 2:
        stat_kw, pval_kw = stats.kruskal(*grupos_dieta)
        sig_kw = "** Significativo" if pval_kw < 0.05 else "Não significativo"
        print(f"\n  2. Kruskal-Wallis: tipo de dieta vs {TARGET}")
        print(f"     H={stat_kw:.3f},  p={pval_kw:.4f}  →  {sig_kw} (alpha=0.05)")
        resumo_dieta = df.groupby("diet_type")[TARGET].agg(["mean", "median", "count"]).round(3)
        print(resumo_dieta.to_string())

# Teste 3: Correlação de Spearman entre experiência do nutricionista e resultado
# Usa-se Spearman (robusto a outliers) em vez de Pearson
if "years_experience" in df.columns and TARGET in df.columns:
    data_exp = df[["years_experience", TARGET]].dropna()
    rho, pval_sp = stats.spearmanr(data_exp["years_experience"], data_exp[TARGET])
    sig_sp = "** Significativo" if pval_sp < 0.05 else "Não significativo"
    print(f"\n  3. Correlação de Spearman: years_experience vs {TARGET}")
    print(f"     rho={rho:.4f},  p={pval_sp:.4f}  →  {sig_sp} (alpha=0.05)")

# Teste 4: Diferença entre abordagens do nutricionista (Kruskal-Wallis)
if "approach" in df.columns and TARGET in df.columns:
    grupos_abord = [g[TARGET].dropna().values
                    for _, g in df.groupby("approach")
                    if len(g) >= 5]
    if len(grupos_abord) >= 2:
        stat_ab, pval_ab = stats.kruskal(*grupos_abord)
        sig_ab = "** Significativo" if pval_ab < 0.05 else "Não significativo"
        print(f"\n  4. Kruskal-Wallis: abordagem do nutricionista vs {TARGET}")
        print(f"     H={stat_ab:.3f},  p={pval_ab:.4f}  →  {sig_ab} (alpha=0.05)")

print("═" * 55)

# ── 4. Scatter plots: top features numéricas vs target ────────
print("\n[3/7] A gerar scatter plots (numéricas vs target)...")
top_6 = corr_target.abs().nlargest(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
for i, feat in enumerate(top_6):
    ax = axes[i]
    data = df[[feat, TARGET]].dropna()
    ax.scatter(data[feat], data[TARGET], alpha=0.25, s=15, color="steelblue")
    z = np.polyfit(data[feat], data[TARGET], 1)
    x_line = np.linspace(data[feat].min(), data[feat].max(), 200)
    ax.plot(x_line, np.poly1d(z)(x_line), "r--", linewidth=2, label="Tendência")
    r = corr_target.get(feat, np.nan)
    ax.set_title(f"{feat}  (r = {r:.3f})", fontsize=10, fontweight="bold")
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel(TARGET, fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.legend(fontsize=8)

plt.suptitle("Scatter Plots: Top 6 Variáveis Numéricas vs Variável-Alvo", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{GRAFICOS_DIR}/bivariada_scatter_target.png", bbox_inches="tight")
plt.close()
print(f"   Guardado: {GRAFICOS_DIR}/bivariada_scatter_target.png")

# ── 5. Box plots: target por variável categórica ──────────────
print("[4/7] A gerar box plots (target por cada variável categórica)...")
cats_para_boxplot = [c for c in categ_cols if c not in ["estacao", "mes"]]
n_cats = len(cats_para_boxplot)

if n_cats > 0:
    fig, axes = plt.subplots(1, n_cats, figsize=(7 * n_cats, 7))
    if n_cats == 1:
        axes = [axes]
    for i, col in enumerate(cats_para_boxplot):
        order = df.groupby(col)[TARGET].median().sort_values().index
        sns.boxplot(x=col, y=TARGET, data=df, ax=axes[i], order=order, palette="muted")
        axes[i].axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
        axes[i].set_title(f"'{TARGET}'\npor '{col}'", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Variação de Peso (kg)")
        axes[i].tick_params(axis="x", rotation=35)
        means = df.groupby(col)[TARGET].mean()
        y_off = df[TARGET].min() - df[TARGET].std() * 0.15
        for j, cat in enumerate(order):
            axes[i].text(j, y_off, f"μ={means[cat]:.2f}",
                         ha="center", fontsize=8, color="navy", fontweight="bold")
    plt.suptitle("Box Plots: Variação de Peso por Variável Categórica", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{GRAFICOS_DIR}/bivariada_boxplot_categoricas.png", bbox_inches="tight")
    plt.close()
    print(f"   Guardado: {GRAFICOS_DIR}/bivariada_boxplot_categoricas.png")

# ── 6. Análise sazonal: variação de peso por estação ──────────
# Aborda o padrão do enunciado: "dietas em meses frios vs quentes"
print("[5/7] A gerar análise sazonal (estação vs resultado)...")
if "estacao" in df.columns:
    ordem = ["inverno", "primavera", "verao", "outono"]
    ordem_disp = [e for e in ordem if e in df["estacao"].values]
    cores_est = {"inverno": "#4e9af1", "primavera": "#a8d8a8", "verao": "#ffcf77", "outono": "#e88c7d"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot: peso perdido por estação
    sns.boxplot(x="estacao", y=TARGET, data=df, order=ordem_disp,
                palette=cores_est, ax=axes[0])
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[0].set_title("Variação de Peso por Estação do Ano", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Estação")
    axes[0].set_ylabel("Variação de Peso (kg)")
    medias = df.groupby("estacao")[TARGET].mean()
    for j, est in enumerate(ordem_disp):
        axes[0].text(j, df[TARGET].min() - 0.3, f"μ={medias[est]:.2f}",
                     ha="center", fontsize=9, color="navy", fontweight="bold")

    # Taxa de sucesso por estação + faixa etária (<50 vs >=50)
    if "age" in df.columns:
        df["faixa_etaria"] = df["age"].apply(lambda a: "<50 anos" if a < 50 else "≥50 anos")
        sucesso_cruzado = df.groupby(["estacao", "faixa_etaria"])["sucesso"].mean().unstack() * 100
        sucesso_cruzado = sucesso_cruzado.reindex(ordem_disp)
        sucesso_cruzado.plot(kind="bar", ax=axes[1], colormap="Set2", edgecolor="white")
        axes[1].set_title("Taxa de Sucesso por Estação e Faixa Etária", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Estação")
        axes[1].set_ylabel("Taxa de Sucesso (%)")
        axes[1].set_ylim(0, 100)
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].legend(title="Faixa Etária")

    plt.suptitle("Padrões Sazonais no Sucesso das Dietas", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{GRAFICOS_DIR}/bivariada_sazonalidade.png", bbox_inches="tight")
    plt.close()
    print(f"   Guardado: {GRAFICOS_DIR}/bivariada_sazonalidade.png")

# ── 7. Cross-análise: abordagem do nutricionista × sexo ───────
# Aborda o padrão do enunciado: "nutricionistas mais exigentes têm mais pacientes masculinos"
print("[6/7] A gerar análise abordagem do nutricionista × sexo do paciente...")
if "approach" in df.columns and "sex" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap de proporção (% de cada sexo dentro de cada abordagem)
    cross = pd.crosstab(df["approach"], df["sex"], normalize="index") * 100
    sns.heatmap(cross, annot=True, fmt=".1f", cmap="Blues", ax=axes[0],
                cbar_kws={"label": "% de pacientes (dentro da abordagem)"},
                annot_kws={"size": 11})
    axes[0].set_title("% de Pacientes por Sexo\ndentro de cada Abordagem do Nutricionista",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Sexo do Paciente")
    axes[0].set_ylabel("Abordagem do Nutricionista")

    # Taxa de sucesso por abordagem e sexo
    suc = df.groupby(["approach", "sex"])["sucesso"].mean().unstack() * 100
    suc.plot(kind="bar", ax=axes[1], colormap="Set1", edgecolor="white")
    axes[1].set_title("Taxa de Sucesso por Abordagem do Nutricionista e Sexo",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Abordagem do Nutricionista")
    axes[1].set_ylabel("Taxa de Sucesso (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(title="Sexo")

    plt.suptitle("Influência da Abordagem do Nutricionista e do Sexo no Sucesso da Dieta",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{GRAFICOS_DIR}/bivariada_abordagem_sexo.png", bbox_inches="tight")
    plt.close()
    print(f"   Guardado: {GRAFICOS_DIR}/bivariada_abordagem_sexo.png")

    print("\n  Distribuição de sexo por abordagem do nutricionista (%):")
    print(cross.round(1).to_string())

# ── 8. Taxa de sucesso por tipo de dieta ──────────────────────
print("[7/7] A gerar taxa de sucesso por tipo de dieta...")
if "diet_type" in df.columns and "sucesso" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Taxa de sucesso por tipo de dieta
    suc_dieta = df.groupby("diet_type")["sucesso"].mean().sort_values(ascending=False) * 100
    bars = axes[0].bar(suc_dieta.index, suc_dieta.values,
                       color=sns.color_palette("muted", len(suc_dieta)))
    axes[0].set_title("Taxa de Sucesso por Tipo de Dieta", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Taxa de Sucesso (%)")
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, suc_dieta.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    # Variação de peso média por tipo de dieta
    media_dieta = df.groupby("diet_type")[TARGET].mean().sort_values()
    bars2 = axes[1].barh(media_dieta.index, media_dieta.values,
                         color=["steelblue" if v < 0 else "salmon" for v in media_dieta.values])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Variação de Peso Média por Tipo de Dieta", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Variação média de peso (kg)")
    for bar, val in zip(bars2, media_dieta.values):
        offset = 0.02 if val >= 0 else -0.02
        axes[1].text(val + offset, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f} kg", ha="left" if val >= 0 else "right", va="center", fontsize=9)

    plt.suptitle("Eficácia dos Tipos de Dieta", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{GRAFICOS_DIR}/bivariada_sucesso_dieta.png", bbox_inches="tight")
    plt.close()
    print(f"   Guardado: {GRAFICOS_DIR}/bivariada_sucesso_dieta.png")

# ── Tabela-resumo ──────────────────────────────────────────────
print("\n" + "═" * 55)
print("  RESUMO: Média de variação de peso por grupo")
print("═" * 55)
for col in [c for c in categ_cols if c != "estacao"]:
    summary = (df.groupby(col)[TARGET]
               .agg(["mean", "median", "count"])
               .rename(columns={"mean": "Média (kg)", "median": "Mediana (kg)", "count": "N"})
               .round(3)
               .sort_values("Média (kg)"))
    print(f"\n  Por '{col}':")
    print(summary.to_string())

print(f"\n✓ AED Bivariada concluída. Gráficos guardados em: {GRAFICOS_DIR}/")
