import os, sys

# --- Path bootstrap so 'src.*' imports work regardless of CWD ---
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt

from src.plotting import metric_vs_k, top_mi_bar
from src.data import load_wdbc
from src.eval_protocol import make_classifier, make_selector

OUTDIR = os.path.join(PROJECT_ROOT, "outputs")


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    results = pd.read_csv(os.path.join(OUTDIR, "results_cv.csv"))

    # Curves vs k (averaged over classifiers to declutter)
    for metric in ["accuracy", "roc_auc"]:
        df = (
            results
            .groupby(["selector", "k", "fold_id"], as_index=False)[metric]
            .mean()
            .loc[:, ["selector", "k", metric]]
        )
        metric_vs_k(df, metric=metric, outpath=os.path.join(OUTDIR, f"fig_{metric}_vs_k.png"),
                    title=f"{metric.upper()} vs k (mean ± 95% CI)")

    # MI bar chart (top 10)
    mi = pd.read_csv(os.path.join(OUTDIR, "mi_scores.csv")).sort_values("mi", ascending=False)
    top_mi_bar(mi["mi"].values, mi["feature"].values, k=10, outpath=os.path.join(OUTDIR, "fig_top_mi.png"))

    # ROC curves at best-k for a fixed classifier (LogReg) across a few selectors
    X, y, feat_names, target_names = load_wdbc()
    summary = pd.read_csv(os.path.join(OUTDIR, "summary_by_method.csv"))
    clf_name = "LogReg"
    sels = ["MI", "mRMR", "FScore", "L1LogReg", "RFImp"]

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=42,
                                                        stratify=y.values)

    plt.figure(figsize=(7, 5))
    for sel_name in sels:
        row = summary[(summary.selector == sel_name) & (summary.classifier == clf_name)]
        if row.empty:
            continue
        best_k = int(row.iloc[0]["best_k"])
        selector = make_selector(sel_name, best_k)
        clf = make_classifier(clf_name)
        pipe = Pipeline([("selector", selector), ("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_score = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
            smin, smax = scores.min(), scores.max()
            y_score = (scores - smin) / (smax - smin + 1e-12)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        RocCurveDisplay(fpr=fpr, tpr=tpr, name=f"{sel_name} (k={best_k})").plot(ax=plt.gca())

    plt.title("ROC curves @ best-k (LogReg)", fontsize=16, fontweight="bold")
    plt.savefig(os.path.join(OUTDIR, "fig_roc_bestk.png"), dpi=200, bbox_inches="tight")
    plt.xlim(left=0.0, right=1.0)
    plt.grid(True, linestyle="--", linewidth=1, alpha=0.7)
    plt.close()

    print("Figures written to ./outputs")


if __name__ == "__main__":
    main()
