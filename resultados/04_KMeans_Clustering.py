"""
=============================================================
  SPRINT 2 - K-MEANS CLUSTERING
  Membro 1: Elbow Method + K-Means + Visualização PCA
  Membro 2: Interpretação dos Clusters em linguagem de negócio
=============================================================
  Input  : resultados/dataset_limpo.csv
  Output : resultados/graficos/kmeans_*.png
           resultados/clusters_perfis.csv
  Executar da raiz do projeto: python resultados/04_KMeans_Clustering.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ── Configurações visuais ──────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── Caminhos ───────────────────────────────────────────────────
INPUT_PATH   = "resultados/dataset_limpo.csv"
GRAFICOS_DIR = "resultados/graficos"
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# ── 1. Carregar e enriquecer dados ─────────────────────────────
df_orig = pd.read_csv(INPUT_PATH)
print(f"Dataset: {df_orig.shape[0]} linhas × {df_orig.shape[1]} colunas")

# Extrair estação do ano (feature sazonal para clustering)
# Permite encontrar padrões como "pacientes no inverno perdem mais peso"
if "record_created_at" in df_orig.columns:
    df_orig["mes"] = pd.to_datetime(df_orig["record_created_at"], errors="coerce").dt.month
    def mes_para_estacao(m):
        if m in [12, 1, 2]:  return "inverno"
        elif m in [3, 4, 5]: return "primavera"
        elif m in [6, 7, 8]: return "verao"
        else:                 return "outono"
    df_orig["estacao"] = df_orig["mes"].map(mes_para_estacao)

# Variável de sucesso (para análise pós-clustering)
TARGET = "weight_change_kg_6m"
if TARGET in df_orig.columns:
    df_orig["sucesso"] = (df_orig[TARGET] < 0).astype(int)

# ── 2. Selecionar features para clustering ─────────────────────
# Apenas features do paciente/plano ANTES dos resultados.
# Variáveis de outcome ficam de fora para que os clusters representem
# perfis de entrada, não o que já aconteceu.
COLS_EXCLUIR_CLUSTER = [
    "program_id", "patient_id", "nutritionist_id", "diet_id",
    "record_created_at", "program_index",
    "weight_change_kg_6m",      # outcome — variável-alvo
    "mean_adherence_pct",       # outcome — medido após o programa
    "adherence_ratio",          # outcome — derivado de mean_adherence_pct
    "motivation_score_program", # outcome — medido durante o programa
    "sucesso",                  # derivado do outcome
    "mes",                      # redundante com estacao
]

df_cluster = df_orig.drop(columns=[c for c in COLS_EXCLUIR_CLUSTER if c in df_orig.columns]).copy()

# Normalizar e codificar categóricas (inclui estacao)
categ_cols_c = df_cluster.select_dtypes(include="object").columns.tolist()
for col in categ_cols_c:
    df_cluster[col] = df_cluster[col].str.strip().str.lower()

df_encoded = pd.get_dummies(df_cluster, columns=categ_cols_c, drop_first=True, dtype=float)

# Escalar todas as features (K-Means é sensível à escala)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)
feature_names = df_encoded.columns.tolist()

print(f"Features para clustering ({len(feature_names)}): {feature_names}")


# ═══════════════════════════════════════════════════════════════
# MEMBRO 1: ELBOW METHOD + K-MEANS + VISUALIZAÇÃO PCA
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  MEMBRO 1: Elbow Method, K-Means e Visualização PCA")
print("═" * 60)

# ── 3. Elbow Method + Silhouette Score ─────────────────────────
print("\n[1/3] A calcular inércia e Silhouette Score para K=1..10...")
K_RANGE          = range(1, 11)
inercias          = []
silhouette_scores = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X_scaled)
    inercias.append(km.inertia_)
    if k >= 2:
        sample = min(1000, X_scaled.shape[0])
        sil = silhouette_score(X_scaled, km.labels_, sample_size=sample, random_state=42)
        silhouette_scores.append(sil)
        print(f"   K={k:2d} → inércia={km.inertia_:10.1f}   silhouette={sil:.4f}")
    else:
        print(f"   K={k:2d} → inércia={km.inertia_:10.1f}   silhouette=N/A")

K_OTIMO = list(range(2, 11))[silhouette_scores.index(max(silhouette_scores))]
print(f"\n  → K ótimo pelo Silhouette Score: K = {K_OTIMO}  (score = {max(silhouette_scores):.4f})")

# Plot: Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(K_RANGE), inercias, "bo-", linewidth=2, markersize=8)
axes[0].set_title("Elbow Method — Inércia por Número de Clusters", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Número de Clusters (K)")
axes[0].set_ylabel("Inércia (SSE)")
axes[0].set_xticks(list(K_RANGE))
axes[0].grid(True, alpha=0.3)

k_sil = list(range(2, 11))
axes[1].plot(k_sil, silhouette_scores, "ro-", linewidth=2, markersize=8)
axes[1].axvline(K_OTIMO, color="green", linestyle="--", linewidth=2, label=f"K ótimo = {K_OTIMO}")
axes[1].set_title("Silhouette Score por Número de Clusters", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Número de Clusters (K)")
axes[1].set_ylabel("Silhouette Score  (maior = melhor)")
axes[1].set_xticks(k_sil)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Determinação do K Ótimo para K-Means", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{GRAFICOS_DIR}/kmeans_elbow_silhouette.png", bbox_inches="tight")
plt.close()
print(f"   Guardado: {GRAFICOS_DIR}/kmeans_elbow_silhouette.png")

# ── 4. K-Means com K ótimo ────────────────────────────────────
print(f"\n[2/3] A treinar K-Means com K={K_OTIMO}...")
km_final = KMeans(n_clusters=K_OTIMO, random_state=42, n_init=10, max_iter=300)
km_final.fit(X_scaled)
labels = km_final.labels_

df_orig["cluster"] = labels

print("\n  Distribuição dos clusters:")
dist = df_orig["cluster"].value_counts().sort_index()
for k, n in dist.items():
    print(f"    Cluster {k}: {n} programas  ({n/len(df_orig)*100:.1f}%)")

# ── 5. Visualização PCA 2D ─────────────────────────────────────
print(f"\n[3/3] A gerar visualização PCA 2D...")
pca     = PCA(n_components=2, random_state=42)
X_pca   = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_
palette = sns.color_palette("Set2", K_OTIMO)

fig, ax = plt.subplots(figsize=(10, 7))
for k in range(K_OTIMO):
    mask = labels == k
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[palette[k]], label=f"Cluster {k}  (n={mask.sum()})",
               alpha=0.45, s=25, edgecolors="none")

centroids_pca = pca.transform(km_final.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           marker="X", s=200, c="black", zorder=5, label="Centroides")

ax.set_title(f"K-Means K={K_OTIMO} — Visualização PCA 2D", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1  ({var_exp[0]*100:.1f}% variância)")
ax.set_ylabel(f"PC2  ({var_exp[1]*100:.1f}% variância)")
ax.legend(markerscale=1.5, fontsize=9)
ax.grid(True, alpha=0.25)
ax.text(0.02, 0.02, f"Variância total explicada: {(var_exp[0]+var_exp[1])*100:.1f}%",
        transform=ax.transAxes, fontsize=9, color="gray")

plt.tight_layout()
plt.savefig(f"{GRAFICOS_DIR}/kmeans_pca_clusters.png", bbox_inches="tight")
plt.close()
print(f"   Guardado: {GRAFICOS_DIR}/kmeans_pca_clusters.png")


# ═══════════════════════════════════════════════════════════════
# COMPARAÇÃO: CLUSTERING HIERÁRQUICO vs K-MEANS
# O enunciado pede "técnicas de clustering" (plural) — testamos dois métodos
# e justificamos a escolha com base no Silhouette Score.
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  COMPARAÇÃO: Clustering Hierárquico (Ward) vs K-Means")
print("═" * 60)

# ── 1. Clustering Hierárquico com mesmo K que o K-Means ─────────────────
print(f"\n[a] A treinar Clustering Hierárquico com K={K_OTIMO} (link=ward)...")
hier = AgglomerativeClustering(n_clusters=K_OTIMO, linkage="ward")
labels_hier = hier.fit_predict(X_scaled)

# Silhouette Scores comparativos
sample_n  = min(2000, X_scaled.shape[0])
sil_kmeans = silhouette_score(X_scaled, labels,       sample_size=sample_n, random_state=42)
sil_hier   = silhouette_score(X_scaled, labels_hier,  sample_size=sample_n, random_state=42)

print(f"   Silhouette Score — K-Means       : {sil_kmeans:.4f}")
print(f"   Silhouette Score — Hierárquico   : {sil_hier:.4f}")
melhor_metodo = "K-Means" if sil_kmeans >= sil_hier else "Hierárquico"
print(f"   → Melhor método (maior silhouette): {melhor_metodo}")

# ── 2. Dendrograma (amostra de até 500 pontos para legibilidade) ─────────
print(f"\n[b] A gerar dendrograma (amostra de 200 pontos)...")
np.random.seed(42)
amostra_idx = np.random.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
X_dendro    = X_scaled[amostra_idx]

Z = linkage(X_dendro, method="ward")

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(
    Z, ax=ax,
    truncate_mode="lastp",  # mostra os últimos p merged clusters
    p=20,
    leaf_rotation=90,
    leaf_font_size=9,
    color_threshold=Z[-K_OTIMO, 2],  # marca o corte no número de clusters
)
ax.axhline(y=Z[-K_OTIMO, 2], color="red", linestyle="--", linewidth=1.5,
           label=f"Corte para K={K_OTIMO} clusters")
ax.set_title(f"Dendrograma — Clustering Hierárquico (Ward)\nAmostra de {min(200, len(X_scaled))} observações",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Observação")
ax.set_ylabel("Distância Ward")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{GRAFICOS_DIR}/hier_dendrograma.png", bbox_inches="tight")
plt.close()
print(f"   Guardado: {GRAFICOS_DIR}/hier_dendrograma.png")

# ── 3. Gráfico comparativo PCA: K-Means vs Hierárquico ────────────────
print(f"\n[c] A gerar comparação visual PCA 2D...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

palette_k = sns.color_palette("Set2", K_OTIMO)
palette_h = sns.color_palette("Set1", K_OTIMO)

for ax, (lbl, nome, pal, sil_val) in zip(
    axes,
    [
        (labels,      f"K-Means (silhouette={sil_kmeans:.3f})",      palette_k, sil_kmeans),
        (labels_hier, f"Hierárquico Ward (silhouette={sil_hier:.3f})", palette_h, sil_hier),
    ],
):
    for k in range(K_OTIMO):
        mask = lbl == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[pal[k]], label=f"Cluster {k}  (n={mask.sum()})",
                   alpha=0.45, s=20, edgecolors="none")
    ax.set_title(nome, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var.)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var.)", fontsize=9)
    ax.legend(markerscale=1.5, fontsize=8)
    ax.grid(True, alpha=0.25)

fig.suptitle(f"Comparação: K-Means vs Clustering Hierárquico (K={K_OTIMO})\n"
             f"Método selecionado: {melhor_metodo} (maior Silhouette Score)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{GRAFICOS_DIR}/comparacao_kmeans_hier.png", bbox_inches="tight")
plt.close()
print(f"   Guardado: {GRAFICOS_DIR}/comparacao_kmeans_hier.png")
print(f"\n✓ Comparação de métodos concluída. Método final: {melhor_metodo}")


# ═══════════════════════════════════════════════════════════════
# MEMBRO 2: INTERPRETAÇÃO DOS CLUSTERS
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  MEMBRO 2: Interpretação dos Clusters")
print("═" * 60)

# ── 6. Perfil médio de cada cluster ───────────────────────────
print("\n[1/3] A calcular perfis dos clusters...")
FEATS_INTERP = ["age", "baseline_weight_kg", "baseline_bmi", "sleep_hours",
                "motivation_score", "years_experience",
                "carb_pct", "protein_pct", "fat_pct",
                "mean_adherence_pct", TARGET]
feats_disp = [f for f in FEATS_INTERP if f in df_orig.columns]

perfis_num = df_orig.groupby("cluster")[feats_disp].mean().round(3)
print("\nPerfil numérico médio por cluster:")
print(perfis_num.T.to_string())
perfis_num.to_csv("resultados/clusters_perfis.csv")
print("\n   Guardado: resultados/clusters_perfis.csv")

# Distribuição categórica por cluster
categ_orig = [c for c in df_orig.select_dtypes(include="object").columns
              if c not in ["cluster"]]
for col in categ_orig:
    df_orig[col] = df_orig[col].str.strip().str.lower()
    print(f"\n  '{col}' por cluster (% dentro do cluster):")
    cross = pd.crosstab(df_orig["cluster"], df_orig[col], normalize="index").round(3) * 100
    print(cross.to_string())

# Distribuição da estação do ano por cluster (se disponível)
if "estacao" in df_orig.columns:
    print("\n  'estacao' por cluster (%):")
    cross_est = pd.crosstab(df_orig["cluster"], df_orig["estacao"], normalize="index").round(3) * 100
    print(cross_est.to_string())

# ── 7. Heatmap dos perfis ─────────────────────────────────────
print("\n[2/3] A gerar heatmap dos perfis dos clusters...")
perfis_norm = perfis_num.apply(lambda col: (col - col.mean()) / (col.std() + 1e-8), axis=0)

fig, ax = plt.subplots(figsize=(max(8, K_OTIMO * 2), max(6, len(feats_disp) * 0.6)))
sns.heatmap(
    perfis_norm.T, annot=perfis_num.T.round(1), fmt="g",
    cmap="RdBu_r", center=0, ax=ax, linewidths=0.5,
    annot_kws={"size": 9},
    cbar_kws={"label": "Desvio da média entre clusters (z-score)"},
)
ax.set_title(f"Perfil dos {K_OTIMO} Clusters — Valores médios reais (cor = desvio relativo)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Cluster")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{GRAFICOS_DIR}/kmeans_heatmap_perfis.png", bbox_inches="tight")
plt.close()
print(f"   Guardado: {GRAFICOS_DIR}/kmeans_heatmap_perfis.png")

# ── 8. Taxa de sucesso e resultado médio por cluster ──────────
print("\n[3/3] A gerar gráfico de sucesso por cluster...")
if "sucesso" in df_orig.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    suc_cluster = df_orig.groupby("cluster")["sucesso"].mean().sort_index() * 100
    bars = axes[0].bar(suc_cluster.index.astype(str), suc_cluster.values,
                       color=palette[:K_OTIMO])
    axes[0].set_title("Taxa de Sucesso por Cluster", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Taxa de Sucesso (%)")
    axes[0].set_ylim(0, 100)
    for bar, val in zip(bars, suc_cluster.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    peso_cluster = df_orig.groupby("cluster")[TARGET].mean().sort_index()
    colors_b = ["steelblue" if v < 0 else "salmon" for v in peso_cluster.values]
    bars2 = axes[1].bar(peso_cluster.index.astype(str), peso_cluster.values, color=colors_b)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Variação de Peso Média por Cluster", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Variação média de peso (kg)")
    for bar, val in zip(bars2, peso_cluster.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     val + (0.05 if val >= 0 else -0.15),
                     f"{val:+.2f} kg", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Resultados por Cluster", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{GRAFICOS_DIR}/kmeans_sucesso_clusters.png", bbox_inches="tight")
    plt.close()
    print(f"   Guardado: {GRAFICOS_DIR}/kmeans_sucesso_clusters.png")

# ── 9. Interpretação em linguagem de negócio ─────────────────
print("\n" + "═" * 60)
print("  INTERPRETAÇÃO DOS CLUSTERS — LINGUAGEM DE NEGÓCIO")
print("═" * 60)

for k in range(K_OTIMO):
    g   = df_orig[df_orig["cluster"] == k]
    n   = len(g)

    peso_med   = g[TARGET].mean()                                if TARGET          in g else None
    bmi_med    = g["baseline_bmi"].mean()                        if "baseline_bmi"  in g else None
    idade_med  = g["age"].mean()                                  if "age"           in g else None
    ader_med   = g["mean_adherence_pct"].mean()                  if "mean_adherence_pct" in g else None
    motiv_med  = g["motivation_score"].mean()                    if "motivation_score"   in g else None
    sono_med   = g["sleep_hours"].mean()                          if "sleep_hours"   in g else None
    top_dieta  = g["diet_type"].mode()[0]                        if "diet_type"     in g.columns else "N/A"
    top_abord  = g["approach"].mode()[0]                         if "approach"      in g.columns else "N/A"
    top_est    = g["estacao"].mode()[0]                          if "estacao"       in g.columns else "N/A"

    print(f"\n  ── Cluster {k}  (n={n}, {n/len(df_orig)*100:.1f}% do total) ──")
    if idade_med:  print(f"    Idade média:             {idade_med:.1f} anos")
    if bmi_med:    print(f"    IMC médio (entrada):     {bmi_med:.1f}")
    if sono_med:   print(f"    Horas de sono médias:    {sono_med:.1f} h")
    if motiv_med:  print(f"    Score de motivação:      {motiv_med:.1f}")
    if ader_med:   print(f"    Aderência ao programa:   {ader_med:.1f}%")
    print(f"    Tipo de dieta mais comum:    {top_dieta}")
    print(f"    Abordagem nutricionista:     {top_abord}")
    print(f"    Estação mais comum:          {top_est}")

    if peso_med is not None:
        print(f"    RESULTADO MÉDIO:         {peso_med:+.2f} kg em 6 meses")
        if   peso_med < -4:   classif = "Alto sucesso — perda de peso significativa"
        elif peso_med < -1.5: classif = "Sucesso moderado — perda consistente"
        elif peso_med < 0:    classif = "Sucesso leve — perda marginal"
        elif peso_med < 1.5:  classif = "Estável — sem mudança relevante"
        else:                 classif = "Insucesso — ganho de peso"
        print(f"    → Classificação:         {classif}")

print(f"\n✓ Clustering concluído. Gráficos guardados em: {GRAFICOS_DIR}/")
print("  Perfis dos clusters: resultados/clusters_perfis.csv")
