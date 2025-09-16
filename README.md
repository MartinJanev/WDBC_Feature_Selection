# MI vs. Standard Feature Selection on WDBC

This project systematically compares **mutual information (MI)**--based
feature selection (univariate MI and redundancy-aware mRMR) against
**standard selectors** (variance threshold, ANOVA F-score, L1-logistic,
tree-based importance, recursive feature elimination, and PCA as a
dimensionality baseline) on the **Breast Cancer Wisconsin (Diagnostic)**
dataset (WDBC; 569 samples, 30 features).

------------------------------------------------------------------------

## Key ideas

-   **Mutual Information (MI):** Quantifies reduction in uncertainty
    about the label given a feature; captures non-linear dependencies.
-   **mRMR:** Minimum Redundancy--Maximum Relevance. Greedy forward
    selection maximizing\
    \[ I(X_i;Y) -
    `\lambda `{=tex}`\cdot `{=tex}`\frac{1}{|S|}`{=tex}`\sum`{=tex}\_{j
    `\in `{=tex}S} I(X_i;X_j), \] balancing relevance against redundancy
    among already-selected features.
-   **Baselines:** Cover filter, wrapper, and embedded families
    (variance, F-score, L1-logistic, RF importance, RFE, PCA).

------------------------------------------------------------------------

## What you get

-   **Cross-validation results:** Stratified 5×5 repeated folds with
    metrics vs. k.
-   **Significance testing:** Wilcoxon signed-rank comparisons between
    MI/mRMR and baselines across k.
-   **Efficiency & stability analyses:**
    -   *k@95% efficiency* (min number of features to achieve ≥95% of
        own peak accuracy).
    -   *Stability:* Jaccard overlap of selected feature sets.
-   **Outputs:**
    -   CSVs with fold-level metrics, summaries, and stability indices.
    -   Publication-style figures: accuracy/ROC-AUC vs k, ROC at best-k,
        MI bar charts, paired-difference bars, variability and stability
        plots.

------------------------------------------------------------------------

## Repository structure

    ├── src/
    │   ├── data.py              # load_wdbc dataset loader
    │   ├── eval_protocol.py     # CV evaluation logic, selectors + classifiers
    │   ├── plotting.py          # centralized plotting utilities
    │   ├── selectors/           # MISelector, MRMRSelector, and standard selectors
    │   └── utils.py             # seed setting, CI helpers
    │
    ├── scripts/
    │   ├── run_experiments.py   # run CV, save results/stability CSVs
    │   ├── make_figures.py      # generate publication-ready plots
    │   └── MI_vs_Standard.py    # compare MI vs standard, efficiency/stability tests
    │
    ├── outputs/                 # all generated CSVs and plots
    ├── requirements.txt         # dependencies
    └── README.md                # this file

------------------------------------------------------------------------

## Quickstart

``` bash
# 1. Create environment & install dependencies
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# 2. Run CV experiments (writes ./outputs/results_cv.csv etc.)
python scripts/run_experiments.py

# 3. Generate figures (metric vs k, ROC, MI bars, etc.)
python scripts/make_figures.py

# 4. Compare MI vs Standard selectors (efficiency, paired tests, stability)
python scripts/MI_vs_Standard.py
```

All outputs go to `./outputs/`: - `results_cv.csv` --- fold-wise
metrics\
- `summary_by_method.csv` --- best-k summaries (per selector ×
classifier)\
- `stability.csv` --- Jaccard stability per (selector,k)\
- `mi_scores.csv` --- MI ranking of features\
- `k_efficiency_95.csv` --- efficiency table\
- `paired_tests_smallk.csv` --- statistical comparisons\
- Plots: `fig_accuracy_vs_k.png`, `fig_rocauc_vs_k.png`,
`fig_top_mi.png`, `fig_roc_bestk.png`, plus `fig_paired_diffs_k*.png`,
`fig_variability_*.png`, `fig_stability_smallk.png`


------------------------------------------------------------------------

## Design choices

-   **Dataset:** `sklearn.datasets.load_breast_cancer` (WDBC).\
-   **Selectors implemented in `src/selectors`:**
    -   `MISelector`, `MRMRSelector`
    -   `FScoreSelector`, `VarianceTopKSelector`
    -   `L1LogRegSelector`
    -   `RFImportanceSelector`
    -   `RFELogRegSelector`
    -   PCA baseline
-   **Models:** Logistic Regression, Linear SVM, Random Forest (see
    `src/eval_protocol.py`).\
-   **Evaluation:** Accuracy (primary), ROC-AUC, F1; 95% CIs reported.\
-   **Plotting:** Centralized in `src/plotting.py`; scripts call shared
    functions for consistency.

------------------------------------------------------------------------

## References

-   Guyon & Elisseeff (2003). *An Introduction to Variable and Feature
    Selection*. JMLR.\
-   Peng et al. (2005). *Feature Selection Based on Mutual Information:
    Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy*. IEEE
    TPAMI.\
-   Kraskov et al. (2004). *Estimating Mutual Information*. Phys Rev E.\
-   Wolberg et al. (1992). *Breast Cancer Wisconsin Dataset* (UCI).\
-   Cover & Thomas. *Elements of Information Theory*. Wiley.\
-   scikit-learn documentation for feature selection and classifiers.\
-   Additional resources: surveys and tutorials on MI-based feature
    selection.

**Ethical note:** WDBC is an educational benchmark; results are
methodological and not diagnostic.
