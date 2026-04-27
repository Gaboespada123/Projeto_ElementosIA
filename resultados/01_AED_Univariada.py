"""
=============================================================
  SPRINT 2 - AED UNIVARIADA
  Membro 1: Análise exploratória de variáveis individuais
=============================================================
  Input  : resultados/dataset_limpo.csv
  Output : resultados/graficos/univariada_*.png
  Executar da raiz do projeto: python resultados/01_AED_Univariada.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configurações visuais ──────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── Caminhos e constantes ──────────────────────────────────────
INPUT_PATH   = "resultados/dataset_limpo.csv"
GRAFICOS_DIR = "resultados/graficos"
TARGET       = "weight_change_kg_6m"
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# ── 1. Carregar e enriquecer os dados ─────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")

# Extrair mês e estação do ano de record_created_at
# Necessário para analisar padrões sazonais (ex: "meses frios vs meses quentes")
if "record_created_at" in df.columns:
    df["mes"] = pd.to_datetime(df["record_created_at"], errors="coerce").dt.month

    def mes_para_estacao(m):
        if m in [12, 1, 2]:  return "inverno"
        elif m in [3, 4, 5]: return "primavera"
        elif m in [6, 7, 8]: return "verao"
        else:                 return "outono"

    df["estacao"] = df["mes"].map(mes_para_estacao)

# Criar variável binária de sucesso (perda de peso = sucesso)
if TARGET in df.columns:
    df["sucesso"] = (df[TARGET] < 0).astype(int)

# Remover colunas de ID e data bruta (manter estacao e mes que criámos)
COLS_EXCLUIR = ["program_id", "patient_id", "nutritionist_id", "diet_id", "record_created_at"]
df_work = df.drop(columns=[c for c in COLS_EXCLUIR if c in df.columns])

# Normalizar categóricas
categ_cols_obj = df_work.select_dtypes(include="object").columns.tolist()
for col in categ_cols_obj:
    df_work[col] = df_work[col].str.strip().str.lower()

numeric_cols = df_work.select_dtypes(include="number").columns.tolist()
categ_cols   = df_work.select_dtypes(include="object").columns.tolist()

print(f"Colunas numéricas ({len(numeric_cols)})")
print(f"Colunas categóricas ({len(categ_cols)}): {categ_cols}")
print("\nEstatísticas descritivas:")
print(df_work[numeric_cols].describe().round(2).to_string())

# ── 2. Histogramas das variáveis numéricas ─────────────────────
print("\n[1/5] A gerar histogramas das variáveis numéricas...")
n_cols_grid = 4
n_rows_grid = (len(numeric_cols) + n_cols_grid - 1) // n_cols_grid

fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(20, n_rows_grid * 4))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.histplot(df_work[col].dropna(), kde=True, ax=ax, color="steelblue", edgecolor="white")
    mean_val = df_work[col].mean()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Média: {mean_val:.2f}")
    ax.set_title(col, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Frequência")
    ax.legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("AED Univariada — Distribuição das Variáveis Numéricas", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{GRAFICOS_DIR}/univariada_histogramas.png", bbox_inches="tight")
plt.close()
print(f"   Guardado: {GRAFICOS_DIR}/univariada_histogramas.png")

# ── 3. Boxplots das variáveis numéricas ───────────────────────
print("[2/5] A gerar boxplots das variáveis numéricas...")
fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(20, n_rows_grid * 4))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(y=df_work[col].dropna(), ax=ax, color="lightcoral")
    q1     = df_work[col].quantile(0.25)
    median = df_work[col].median()
    q3     = df_work[col].quantile(0.75)
    ax.set_title(col, fontsize=11, fontweight="bold")
    ax.text(
        0.97, 0.97,
        f"Mediana: {median:.2f}\nQ1: {q1:.2f}\nQ3: {q3:.2f}",
        transform=ax.transAxes, fontsize=7, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("AED Univariada — Boxplots das Variáveis Numéricas", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{GRAFICOS_DIR}/univariada_boxplots.png", bbox_inches="tight")
plt.close()
print(f"   Guardado: {GRAFICOS_DIR}/univariada_boxplots.png")

# ── 4. Análise detalhada da variável-alvo e sucesso ───────────
print("[3/5] A gerar análise da variável-alvo e sucesso...")
if TARGET in df_work.columns:
    y = df_work[TARGET].dropna()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(y, kde=True, ax=axes[0], color="darkorange", edgecolor="white")
    axes[0].axvline(y.mean(),   color="red",  linestyle="--", linewidth=2, label=f"Média: {y.mean():.2f} kg")
    axes[0].axvline(y.median(), color="blue", linestyle="--", linewidth=2, label=f"Mediana: {y.median():.2f} kg")
    axes[0].axvline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.6, label="Sem alteração")
    axes[0].set_title("Histograma + KDE", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Variação de Peso (kg)")
    axes[0].legend(fontsize=8)

    sns.boxplot(y=y, ax=axes[1], color="darkorange")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[1].set_title("Boxplot", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Variação de Peso (kg)")

    sns.violinplot(y=y, ax=axes[2], color="darkorange", inner="box")
    axes[2].axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[2].set_title("Violin Plot", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Variação de Peso (kg)")

    plt.suptitle("Variável-Alvo: weight_change_kg_6m (variação de peso em 6 meses)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{GRAFICOS_DIR}/univariada_variavel_alvo.png", bbox_inches="tight")
    plt.close()
    print(f"   Guardado: {GRAFICOS_DIR}/univariada_variavel_alvo.png")

    lost   = (y < 0).sum()
    gained = (y > 0).sum()
    stable = (y == 0).sum()
    print(f"\n  Estatísticas de weight_change_kg_6m:")
    print(f"    Média:         {y.mean():.3f} kg")
    print(f"    Mediana:       {y.median():.3f} kg")
    print(f"    Desvio Padrão: {y.std():.3f} kg")
    print(f"    Assimetria:    {y.skew():.3f}")
    print(f"    Perderam peso (SUCESSO): {lost}  ({lost/len(y)*100:.1f}%)")
    print(f"    Ganharam peso (INSUCESS): {gained} ({gained/len(y)*100:.1f}%)")
    print(f"    Sem alteração:           {stable} ({stable/len(y)*100:.1f}%)")

# ── 5. Gráficos de barras para variáveis categóricas ──────────
print("[4/5] A gerar gráficos das variáveis categóricas (inclui estação)...")
if categ_cols:
    n_cats = len(categ_cols)
    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 6))
    if n_cats == 1:
        axes = [axes]

    for i, col in enumerate(categ_cols):
        counts = df_work[col].value_counts()
        pct    = (counts / len(df_work) * 100).round(1)
        colors = sns.color_palette("muted", len(counts))

        bars = axes[i].bar(counts.index, counts.values, color=colors)
        axes[i].set_title(f"Distribuição: {col}", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Frequência")
        axes[i].tick_params(axis="x", rotation=35)

        for bar, p in zip(bars, pct):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{p}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    plt.suptitle("AED Univariada — Distribuição das Variáveis Categóricas (inclui estação do ano)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{GRAFICOS_DIR}/univariada_categoricas.png", bbox_inches="tight")
    plt.close()
    print(f"   Guardado: {GRAFICOS_DIR}/univariada_categoricas.png")

# ── 6. Taxa de sucesso por estação do ano ─────────────────────
print("[5/5] A gerar análise de sucesso por estação do ano...")
if "estacao" in df_work.columns and "sucesso" in df_work.columns:
    ordem_estacoes = ["inverno", "primavera", "verao", "outono"]
    estacoes_disp  = [e for e in ordem_estacoes if e in df_work["estacao"].values]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Variação de peso média por estação
    media_est = df_work.groupby("estacao")[TARGET].mean().reindex(estacoes_disp)
    bars = axes[0].bar(media_est.index, media_est.values,
                       color=["#4e9af1", "#a8d8a8", "#ffcf77", "#e88c7d"])
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[0].set_title("Variação de Peso Média por Estação", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Variação média de peso (kg)")
    for bar, val in zip(bars, media_est.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                     f"{val:.2f} kg", ha="center", va="bottom", fontsize=10)

    # Taxa de sucesso (%) por estação
    sucesso_est = df_work.groupby("estacao")["sucesso"].mean().reindex(estacoes_disp) * 100
    bars2 = axes[1].bar(sucesso_est.index, sucesso_est.values,
                        color=["#4e9af1", "#a8d8a8", "#ffcf77", "#e88c7d"])
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Taxa de Sucesso (% de casos com perda de peso) por Estação", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Taxa de sucesso (%)")
    for bar, val in zip(bars2, sucesso_est.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Padrões Sazonais: Sucesso da Dieta por Estação do Ano", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{GRAFICOS_DIR}/univariada_sazonalidade.png", bbox_inches="tight")
    plt.close()
    print(f"   Guardado: {GRAFICOS_DIR}/univariada_sazonalidade.png")

print(f"\n✓ AED Univariada concluída. Gráficos guardados em: {GRAFICOS_DIR}/")
