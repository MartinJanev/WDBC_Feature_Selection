# MI vs. Standard Feature Selection on WDBC

This project compares mutual-information (MI) feature selection (univariate MI and an mRMR-style redundancy-aware variant) to standard baselines (variance top-k, univariate F-score/ANOVA, L1-logistic, tree-based importance, RFE) on the Breast Cancer Wisconsin (Diagnostic) dataset (WDBC; 569 samples, 30 features).

**Key ideas**
- MI measures shared information between a feature and the label (nonparametric, can capture non-linear signal).
- mRMR balances *relevance* (feature–label MI) against *redundancy* (MI among features) via a simple greedy criterion.

**What you get**
- Repeated stratified CV evaluation with metric curves vs. `k` features.
- Summary tables (mean ± 95% CI) for Accuracy / ROC-AUC / F1.
- Stability (Jaccard) of selected features across folds.
- Figures: metric vs. `k`, ROC at best-k, MI bar plots.

---

## Quickstart

```bash
# 1) Create env & install deps
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Run experiments (saves CSVs and some preview plots in ./outputs)
python scripts/run_experiments.py

# 3) Make publication-style figures (reads results from ./outputs)
python scripts/make_figures.py
```

Outputs go to `./outputs/` (created automatically):
- `results_cv.csv` — per-config metrics per fold
- `summary_by_method.csv` — best-k summaries (per selector × classifier)
- `stability.csv` — Jaccard stability by selector & k
- `mi_scores.csv` — global MI scores (full data)
- `fig_accuracy_vs_k.png`, `fig_rocauc_vs_k.png`, `fig_top_mi.png`, `fig_roc_bestk.png`

---

## Design choices

- **Dataset**: `sklearn.datasets.load_breast_cancer()` (WDBC). We standardize *after* selection for modeling; selectors that are scale-sensitive (MI, L1) internally standardize during scoring to avoid distance/penalty artifacts.
- **Selectors**:
  - `MI(k)`: rank by `mutual_info_classif`.
  - `mRMR(k)`: greedy forward selection maximizing `I(X_i;Y) − λ · avg_j I(X_i;X_j)` (feature–feature MI via `mutual_info_regression`).
  - `F-score(k)`: `SelectKBest(f_classif)` equivalent (implemented manually to keep a uniform interface).
  - `VarTopK(k)`: highest sample variance (unsupervised baseline).
  - `L1-LogReg(k)`: train L1 logistic (saga) on standardized data, pick top-`k` by |coef|.
  - `RF-Imp(k)`: rank by `RandomForestClassifier.feature_importances_`.
  - `RFE-LogReg(k)`: recursive feature elimination with L2 logistic.
  - `PCA(k)`: dimensionality-reduction baseline (not feature selection).
- **Models**: Logistic Regression, linear SVM (`SVC(kernel='linear', probability=True)`), Random Forest.
- **Evaluation**: `RepeatedStratifiedKFold` (5 folds × 5 repeats), metrics = Accuracy (primary), ROC-AUC, F1. We sweep `k ∈ {3,5,8,10,15,20,30}` for selectors that use `k`. Best-k chosen by mean Accuracy (change to ROC-AUC in code if preferred). 95% CIs computed as `1.96 * std / sqrt(n)`.

---

## References & background (for intuition)

- Medium – *From Data to Insights: How Mutual Information Revolutionizes Feature Engineering*.
- GeeksforGeeks – *Information Gain and Mutual Information for Machine Learning*.
- Applied Soft Computing (2014) – MI-based feature selection survey/article.
- Stanford NLP IR Book – *Mutual Information* chapter.
- Guyon & Elisseeff (2003) – *An Introduction to Variable and Feature Selection*.
- Peng et al. (2005) – mRMR paper.
- Kraskov et al. (2004) – kNN MI estimator.
- scikit-learn documentation for the algorithms used.

Ethical note: use this dataset for educational research; report methodology transparently and avoid over-claiming generalization beyond the dataset.
