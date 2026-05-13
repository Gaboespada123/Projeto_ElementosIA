# Previsão de Sucesso em Intervenções Nutricionais
### Elementos de Inteligência Artificial e Ciência de Dados — UBI 2025/26

Análise de dados e modelos preditivos aplicados a um dataset de acompanhamento nutricional personalizado, com o objetivo de prever a variação de peso de um paciente ao fim de 6 meses com base no seu perfil, plano alimentar e nutricionista.

---

## Contexto

A eficácia de uma dieta depende de fatores biológicos, comportamentais e clínicos, tornando difícil para profissionais de saúde prever com antecedência quais os planos alimentares com maior probabilidade de sucesso. Este projeto aplica técnicas de Ciência de Dados e IA para descobrir esses padrões e construir modelos preditivos interpretáveis.

---

## Dataset

Quatro ficheiros CSV que, após integração, formam um dataset unificado de **2211 programas nutricionais**:

| Ficheiro | Descrição | Registos |
|---|---|---|
| `data/patients.csv` | Perfil do paciente (idade, IMC, hábitos) | 1 000 |
| `data/diets.csv` | Planos dietéticos (macronutrientes, tipo) | 10 |
| `data/nutritionists.csv` | Perfil do nutricionista (abordagem, especialidade) | 20 |
| `data/outcomes.csv` | Resultados dos programas (variação de peso, aderência) | 2 523 |

**Variável-alvo:** `weight_change_kg_6m` — variação de peso em kg ao fim de 6 meses (negativo = perda de peso).

---

## Pipeline do Projeto

```
  dados brutos (4 CSVs)
         │
         ▼
  ┌─────────────────────┐
  │  1. Integração      │  scripts/limpeza_dados.py
  │     União dos 4 CSV │
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  2. Limpeza         │  scripts/limpeza_dados.py
  │  Nulos · Outliers   │  scripts/preprocessamento_final.py
  │  StandardScaler·OHE │
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  3. AED             │  resultados/01_AED_Univariada.py
  │  Univariada         │  resultados/02_AED_Bivariada.py
  │  Bivariada          │  resultados/sprint2_multivariada_clustering.py
  │  Multivariada + PCA │
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  4. Clustering      │  resultados/04_KMeans_Clustering.py
  │  K-Means · DBSCAN   │  resultados/sprint2_multivariada_clustering.py
  │  Hierárquico        │
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  5. Modelos         │  scripts/modelo_baseline.py
  │  Regressão Linear   │  scripts/modelo_arvores_rf.py
  │  Árvore de Decisão  │  scripts/modelo_xgboost.py
  │  Random Forest      │
  │  XGBoost            │
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  6. Discussão       │  scripts/discussao_critica.py
  │  Crítica + Tabela   │  resultados/tabela_comparativa_todos_modelos.csv
  │  Comparativa        │
  └─────────────────────┘
```

---

## Resultados dos Modelos

### Regressão (previsão de kg perdidos — RMSE, menor é melhor)

| Modelo | Treino | Validação | Teste |
|---|---|---|---|
| Regressão Linear | 0.3078 | 0.5164 | 1.0778 |
| Árvore de Decisão | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

### Classificação (previsão de sucesso/insucesso — Accuracy)

| Modelo | Treino | Validação | Teste |
|---|---|---|---|
| Regressão Logística | 0.9122 | 0.8929 | 0.8959 |
| Árvore de Decisão | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

> Tabela completa com todos os modelos: [`resultados/tabela_comparativa_todos_modelos.csv`](resultados/tabela_comparativa_todos_modelos.csv)

**Divisão dos dados:** 50% treino · 30% validação · 10% teste · 10% reservado

---

## Principais Descobertas

- A **aderência ao plano** (`mean_adherence_pct`) é a variável com maior correlação com a perda de peso
- A **abordagem do nutricionista** influencia a distribuição por sexo dos pacientes — nutricionistas com abordagem *strict* têm proporcionalmente mais pacientes masculinos
- Existem padrões sazonais no sucesso das dietas — programas iniciados no **inverno** apresentam resultados distintos dos iniciados no verão
- O **IMC inicial** e a **motivação do paciente** estão entre as features com maior importância nos modelos de árvore

---

## Estrutura do Repositório

```
Projeto_ElementosIA/
│
├── data/                          # Dados originais (não modificar)
│   ├── patients.csv
│   ├── diets.csv
│   ├── nutritionists.csv
│   └── outcomes.csv
│
├── scripts/                       # Scripts de processamento e modelação
│   ├── limpeza_dados.py           # Sprint 1 — integração e limpeza
│   ├── preprocessamento_final.py  # Sprint 1 — normalização final
│   ├── modelo_baseline.py         # Sprint 3 — Regressão Linear/Logística
│   ├── modelo_arvores_rf.py       # Sprint 3 — Árvore de Decisão + Random Forest
│   ├── modelo_xgboost.py          # Sprint 3 — XGBoost (regressão + classificação)
│   ├── discussao_critica.py       # Sprint 4 — comparação e análise crítica
│   └── previsao_interativa.py     # Previsão interativa para novo paciente
│
├── resultados/                    # Outputs gerados pelos scripts
│   ├── 01_AED_Univariada.py       # Sprint 2 — AED univariada (histogramas, boxplots)
│   ├── 02_AED_Bivariada.py        # Sprint 2 — AED bivariada (correlações, sazonalidade)
│   ├── 04_KMeans_Clustering.py    # Sprint 2 — K-Means + interpretação clusters
│   ├── sprint2_multivariada_clustering.py  # Sprint 2 — PCA + clustering alternativo
│   ├── dataset_unido.csv          # Após integração dos 4 CSVs
│   ├── dataset_limpo.csv          # Após limpeza e remoção de outliers
│   ├── dados_limpos_final.csv     # Dataset final (scaled + encoded)
│   ├── clusters_perfis.csv        # Perfis médios de cada cluster K-Means
│   ├── resultados_modelos_baseline.csv
│   ├── tabela_comparativa_todos_modelos.csv
│   └── graficos/                  # Todos os gráficos gerados
│
├── modelos/                       # Modelos treinados (.pickle)
│   ├── modelo_regressao_linear.pickle
│   ├── modelo_regressao_logistica.pickle
│   ├── modelo_xgb_regressao.pickle
│   └── modelo_xgb_classificacao.pickle
│
├── requirements.txt
├── SETUP.txt
└── README.md
```

---

## Como Executar

### 1. Configurar o ambiente

```bash
conda create -n projeto-elementos python=3.13 -y
conda activate projeto-elementos
pip install -r requirements.txt
```

### 2. Executar por ordem

```bash
# Sprint 1 — Limpeza e pré-processamento
python scripts/limpeza_dados.py
python scripts/preprocessamento_final.py

# Sprint 2 — Análise Exploratória
python resultados/01_AED_Univariada.py
python resultados/02_AED_Bivariada.py
python resultados/sprint2_multivariada_clustering.py
python resultados/04_KMeans_Clustering.py

# Sprint 3 — Modelos preditivos
python scripts/modelo_baseline.py
python scripts/modelo_arvores_rf.py
python scripts/modelo_xgboost.py

# Sprint 4 — Discussão crítica
python scripts/discussao_critica.py
```

### 3. Previsão para um novo paciente

```bash
python scripts/previsao_interativa.py
```

---

## Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| pandas | 2.3.2 | Manipulação de dados |
| numpy | 1.24.3 | Operações numéricas |
| scikit-learn | 1.3.2 | Modelos ML e pré-processamento |
| matplotlib | 3.7.1 | Visualização |
| scipy | 1.11.4 | Análise estatística |

---

## Autores

Trabalho desenvolvido no âmbito da unidade curricular **Elementos de Inteligência Artificial e Ciência de Dados**
Licenciatura em Inteligência Artificial e Ciência de Dados — Universidade da Beira Interior, 2025/26
