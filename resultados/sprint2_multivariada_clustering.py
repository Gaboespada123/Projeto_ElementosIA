"""
=============================================================================
  SPRINT 2 — AED MULTIVARIADO AVANÇADO (PCA) + CLUSTERING ALTERNATIVO
=============================================================================
  Input  : resultados/dados_limpos_final.csv
  Output : resultados/graficos/  (prefixos multivariada_ e clustering_alt_)
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Estilo visual global ──────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE_BINARIO = {0: "#4e9af1", 1: "#f17c4e"}   # azul = insucesso, laranja = sucesso

# ── Pasta de saída ────────────────────────────────────────────────────────────
PASTA_GRAFICOS = os.path.join("resultados", "graficos")
os.makedirs(PASTA_GRAFICOS, exist_ok=True)


# =============================================================================
# 1. CARREGAMENTO E LIMPEZA DEFENSIVA DO CSV
# =============================================================================

caminho_csv = os.path.join("resultados", "dados_limpos_final.csv")
df_raw = pd.read_csv(caminho_csv)

# -- 1a. Normalizar nomes de colunas: remover espaços extras
df_raw.columns = df_raw.columns.str.strip()

# -- 1b. Remover colunas duplicadas que surgem de variantes de maiúsculas/espaços
#        no OHE do sprint anterior (ex: "sex_M ", "sex_m", "sex_female", etc.)
#        Estratégia: consolidar colunas da mesma "família" mantendo apenas
#        a versão canónica (minúsculas, sem espaços).
colunas_renomeadas = {}
for col in df_raw.columns:
    colunas_renomeadas[col] = col.strip().lower().replace(" ", "_")

df_raw = df_raw.rename(columns=colunas_renomeadas)

# Após renomear, podem existir duplicadas — manter a primeira ocorrência de cada nome
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated(keep="first")]

# -- 1c. Converter colunas True/False (strings ou booleanos) para 0/1
for col in df_raw.columns:
    if df_raw[col].dtype == object:
        valores_unicos = df_raw[col].dropna().str.strip().str.lower().unique()
        if set(valores_unicos).issubset({"true", "false"}):
            df_raw[col] = df_raw[col].str.strip().str.lower().map({"true": 1, "false": 0})

# -- 1d. Forçar todas as colunas a numérico (segurança extra)
df_raw = df_raw.apply(pd.to_numeric, errors="coerce")

# -- 1e. Preencher eventuais NaN introduzidos pela conversão com a mediana
df_raw = df_raw.fillna(df_raw.median(numeric_only=True))

print("=" * 62)
print(f"  Dataset carregado: {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas")
print("=" * 62)


# =============================================================================
# 2. SEPARAR A VARIÁVEL ALVO (weight_change_kg_6m)
#    — usada APENAS como rótulo visual, NUNCA para treinar clusters
# =============================================================================

COLUNA_ALVO = "weight_change_kg_6m"

# Extrair e guardar o alvo antes de o remover da matriz de features
weight_change = df_raw[COLUNA_ALVO].copy()

# Variável binária de sucesso: perda de peso = 1 | ganho ou estagnação = 0
# (weight_change_kg_6m < 0  →  perdeu peso  →  sucesso = 1)
sucesso = (weight_change < 0).astype(int)

# Matriz de features: sem a variável alvo
X = df_raw.drop(columns=[COLUNA_ALVO])

print(f"\n  Features para modelação : {X.shape[1]} colunas")
print(f"  Taxa de sucesso (perda de peso): "
      f"{sucesso.mean() * 100:.1f}%  ({sucesso.sum()} / {len(sucesso)} pacientes)")


# =============================================================================
# 3. PCA — VARIÂNCIA EXPLICADA  →  Scree Plot
# =============================================================================

N_COMP_SCREE = 10   # nº de componentes a mostrar no gráfico

pca_completo = PCA(n_components=N_COMP_SCREE, random_state=42)
pca_completo.fit(X)

var_explicada   = pca_completo.explained_variance_ratio_ * 100
var_acumulada   = np.cumsum(var_explicada)
componentes_idx = [f"PC{i+1}" for i in range(N_COMP_SCREE)]

fig, ax1 = plt.subplots(figsize=(9, 5))

# Barras — variância individual
barras = ax1.bar(
    componentes_idx,
    var_explicada,
    color=sns.color_palette("Blues_d", N_COMP_SCREE),
    edgecolor="white",
    linewidth=0.6,
    zorder=2,
)
ax1.set_xlabel("Componente Principal", labelpad=8)
ax1.set_ylabel("Variância Explicada (%)", color="#2c5f8a")
ax1.tick_params(axis="y", labelcolor="#2c5f8a")
ax1.set_ylim(0, var_explicada[0] * 1.35)

# Rótulos numéricos em cima de cada barra
for barra, val in zip(barras, var_explicada):
    ax1.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 0.3,
        f"{val:.1f}%",
        ha="center", va="bottom", fontsize=8.5, color="#2c5f8a",
    )

# Linha — variância acumulada (eixo secundário)
ax2 = ax1.twinx()
ax2.plot(
    componentes_idx, var_acumulada,
    color="#e05c2a", marker="o", linewidth=2, markersize=6, zorder=3,
    label="Variância Acumulada (%)",
)
ax2.axhline(80, color="#e05c2a", linestyle="--", linewidth=0.9, alpha=0.5)
ax2.text(N_COMP_SCREE - 0.4, 81.5, "80%", color="#e05c2a", fontsize=8.5, ha="right")
ax2.set_ylabel("Variância Acumulada (%)", color="#e05c2a")
ax2.tick_params(axis="y", labelcolor="#e05c2a")
ax2.set_ylim(0, 105)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax2.legend(loc="center right", fontsize=9)

ax1.set_title("Scree Plot — Variância Explicada por Componente Principal (PCA)",
              fontsize=12, fontweight="bold", pad=12)
ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

plt.tight_layout()
caminho_scree = os.path.join(PASTA_GRAFICOS, "multivariada_scree_plot.png")
fig.savefig(caminho_scree, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  [✔] Scree Plot guardado → {caminho_scree}")

# Informação útil em consola
n80 = int(np.argmax(var_acumulada >= 80)) + 1
print(f"      Componentes para explicar ≥80% da variância: {n80} PCs")


# =============================================================================
# 4. PCA — VISUALIZAÇÃO 2D  →  Scatter PC1 vs PC2
# =============================================================================

pca_2d = PCA(n_components=2, random_state=42)
X_pca  = pca_2d.fit_transform(X)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Sucesso"] = sucesso.values

rotulos = {0: "Sem perda de peso (insucesso)", 1: "Perda de peso (sucesso)"}

fig, ax = plt.subplots(figsize=(9, 6))

for classe, grupo in df_pca.groupby("Sucesso"):
    ax.scatter(
        grupo["PC1"], grupo["PC2"],
        c=PALETTE_BINARIO[classe],
        label=rotulos[classe],
        alpha=0.55, s=22, edgecolors="none",
    )

# Anotar % de variância em cada eixo
var_pc1 = pca_2d.explained_variance_ratio_[0] * 100
var_pc2 = pca_2d.explained_variance_ratio_[1] * 100
ax.set_xlabel(f"PC1  ({var_pc1:.1f}% variância)", fontsize=10)
ax.set_ylabel(f"PC2  ({var_pc2:.1f}% variância)", fontsize=10)
ax.set_title("PCA 2D — Distribuição por Resultado de Perda de Peso\n"
             "(colorido por variável de sucesso: weight_change_kg_6m < 0)",
             fontsize=11, fontweight="bold", pad=10)

legend = ax.legend(title="Resultado", fontsize=9, title_fontsize=9,
                   framealpha=0.85, edgecolor="grey")
ax.grid(linestyle="--", alpha=0.35)

plt.tight_layout()
caminho_pca2d = os.path.join(PASTA_GRAFICOS, "multivariada_pca_2d_scatter.png")
fig.savefig(caminho_pca2d, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  [✔] PCA 2D Scatter guardado → {caminho_pca2d}")


# =============================================================================
# 5. CLUSTERING HIERÁRQUICO (Agglomerative)
#    — Avaliação com Silhouette Score para k = 2..6
# =============================================================================

INTERVALO_K   = range(2, 7)    # testa k = 2, 3, 4, 5, 6
silhouettes   = {}

print("\n  ── Clustering Hierárquico (Agglomerative) ──────────────────")
print(f"  {'k':>4}   {'Silhouette Score':>18}")
print("  " + "-" * 28)

for k in INTERVALO_K:
    modelo = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = modelo.fit_predict(X)
    score  = silhouette_score(X, labels, sample_size=min(1500, len(X)), random_state=42)
    silhouettes[k] = score
    print(f"  {k:>4}   {score:>18.4f}")

# Melhor k
melhor_k     = max(silhouettes, key=silhouettes.get)
melhor_score = silhouettes[melhor_k]
print(f"\n  ★ Melhor k = {melhor_k}  |  Silhouette Score = {melhor_score:.4f}")

# Guardar os labels do melhor modelo para uso posterior (opcional)
modelo_hierarq_final = AgglomerativeClustering(n_clusters=melhor_k, linkage="ward")
labels_hierarq       = modelo_hierarq_final.fit_predict(X)

# -- Gráfico de barras: Silhouette por k ──────────────────────────────────────
ks      = list(silhouettes.keys())
scores  = list(silhouettes.values())
cores   = ["#f17c4e" if k == melhor_k else "#4e9af1" for k in ks]

fig, ax = plt.subplots(figsize=(8, 5))
barras  = ax.bar(ks, scores, color=cores, edgecolor="white", linewidth=0.7, width=0.55)

for barra, val in zip(barras, scores):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 0.002,
        f"{val:.4f}",
        ha="center", va="bottom", fontsize=9,
    )

ax.set_xlabel("Número de Clusters (k)", fontsize=10)
ax.set_ylabel("Silhouette Score", fontsize=10)
ax.set_xticks(ks)
ax.set_ylim(0, max(scores) * 1.18)
ax.set_title(
    "Clustering Hierárquico — Silhouette Score por Número de Clusters\n"
    f"(★ Melhor: k={melhor_k}, Score={melhor_score:.4f})",
    fontsize=11, fontweight="bold", pad=10,
)
# Legenda manual de cores
from matplotlib.patches import Patch
legenda_items = [
    Patch(color="#f17c4e", label=f"Melhor k = {melhor_k}"),
    Patch(color="#4e9af1", label="Outros valores de k"),
]
ax.legend(handles=legenda_items, fontsize=9, framealpha=0.85)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
caminho_silhouette = os.path.join(PASTA_GRAFICOS, "clustering_alt_silhouette_hierarquico.png")
fig.savefig(caminho_silhouette, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  [✔] Silhouette barras guardado → {caminho_silhouette}")


# =============================================================================
# 6. DBSCAN
# =============================================================================
# Parâmetros padrão ajustados para dados com StandardScaler aplicado:
#   eps       — raio de vizinhança (0.5 é ponto de partida razoável)
#   min_samples — mínimo de pontos para formar um core point
#                 regra prática: 2 × nº de features (limitado a 20)

EPS         = 3.0
MIN_SAMPLES = min(2 * X.shape[1], 20)

dbscan       = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, n_jobs=-1)
labels_db    = dbscan.fit_predict(X)

n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_ruido_db    = int((labels_db == -1).sum())
pct_ruido_db  = n_ruido_db / len(labels_db) * 100

print("\n  ── DBSCAN ──────────────────────────────────────────────────")
print(f"  Parâmetros  : eps={EPS}, min_samples={MIN_SAMPLES}")
print(f"  Clusters encontrados : {n_clusters_db}")
print(f"  Pontos de ruído (outliers) : {n_ruido_db}  ({pct_ruido_db:.1f}% do total)")

if n_clusters_db == 0:
    print("\n  ⚠  DBSCAN não formou clusters com estes parâmetros.")
    print("     Considere aumentar eps ou diminuir min_samples.")
elif n_clusters_db == 1:
    print("\n  ⚠  DBSCAN formou apenas 1 cluster — os dados podem ser muito densos")
    print("     ou eps pode estar demasiado alto. Experimente reduzir eps.")


# =============================================================================
# 7. GRÁFICO BONUS: PCA 2D colorido pelos clusters hierárquicos
#    (permite comparação visual directa com os resultados do K-Means)
# =============================================================================

df_clusters = df_pca.copy()
df_clusters["Cluster_Hierarq"] = labels_hierarq

paleta_clusters = sns.color_palette("Set2", melhor_k)

fig, ax = plt.subplots(figsize=(9, 6))

for cluster_id in sorted(df_clusters["Cluster_Hierarq"].unique()):
    grupo = df_clusters[df_clusters["Cluster_Hierarq"] == cluster_id]
    ax.scatter(
        grupo["PC1"], grupo["PC2"],
        color=paleta_clusters[cluster_id],
        label=f"Cluster {cluster_id + 1}",
        alpha=0.55, s=22, edgecolors="none",
    )

ax.set_xlabel(f"PC1  ({var_pc1:.1f}% variância)", fontsize=10)
ax.set_ylabel(f"PC2  ({var_pc2:.1f}% variância)", fontsize=10)
ax.set_title(
    f"PCA 2D — Clustering Hierárquico (k={melhor_k}, Ward linkage)\n"
    "Espaço reduzido pelos 2 primeiros componentes principais",
    fontsize=11, fontweight="bold", pad=10,
)
ax.legend(title="Cluster", fontsize=9, title_fontsize=9,
          framealpha=0.85, edgecolor="grey")
ax.grid(linestyle="--", alpha=0.35)

plt.tight_layout()
caminho_pca_cluster = os.path.join(PASTA_GRAFICOS,
                                   "clustering_alt_pca2d_hierarquico.png")
fig.savefig(caminho_pca_cluster, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  [✔] PCA 2D Clusters guardado → {caminho_pca_cluster}")


# =============================================================================
# 8. RESUMO FINAL
# =============================================================================

print("\n" + "=" * 62)
print("  RESUMO DO SPRINT 2")
print("=" * 62)
print(f"  PCA — componentes para ≥80% variância  : {n80}")
print(f"  PCA — variância em PC1 + PC2           : {var_pc1 + var_pc2:.1f}%")
print(f"  Clustering Hierárquico — melhor k      : {melhor_k}  "
      f"(Silhouette = {melhor_score:.4f})")
print(f"  DBSCAN — clusters encontrados          : {n_clusters_db}")
print(f"  DBSCAN — pontos de ruído               : {n_ruido_db}  ({pct_ruido_db:.1f}%)")
print("=" * 62)
print(f"\n  Gráficos guardados em: '{PASTA_GRAFICOS}/'")
print("    • multivariada_scree_plot.png")
print("    • multivariada_pca_2d_scatter.png")
print("    • clustering_alt_silhouette_hierarquico.png")
print("    • clustering_alt_pca2d_hierarquico.png")
print()