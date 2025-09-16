import os, sys

# --- Path bootstrap so 'src.*' imports work regardless of CWD ---
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from itertools import product
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt

# New: statistical tests
try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from src.plotting import metric_vs_k, top_mi_bar
from src.WDBC import load_wdbc
from src.eval_protocol import make_classifier, make_selector
from src.stats_sig import run_wilcoxon_grid, write_narrative
from src.plotting import metric_vs_k, top_mi_bar, plot_roc_curves_bestk, plot_pvalues

OUTDIR = os.path.join(PROJECT_ROOT, "outputs")


# ------------------------ NEW: helpers for significance ------------------------
def _paired_vectors(df: pd.DataFrame, sel_a: str, sel_b: str, k: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired vectors (a, b) for a given k and metric aligned by fold and classifier.
    Expects columns: ['selector','k','fold_id','classifier', metric].
    """
    cols = ["fold_id", "classifier"]
    gcols = cols + [metric]
    da = (
        df[(df.selector == sel_a) & (df.k == k)]
        .loc[:, [*cols, metric]]
        .rename(columns={metric: f"{metric}_a"})
    )
    db = (
        df[(df.selector == sel_b) & (df.k == k)]
        .loc[:, [*cols, metric]]
        .rename(columns={metric: f"{metric}_b"})
    )
    merged = pd.merge(da, db, on=cols, how="inner")
    a = merged[f"{metric}_a"].to_numpy()
    b = merged[f"{metric}_b"].to_numpy()
    return a, b


def run_wilcoxon_grid(results: pd.DataFrame,
                      comparisons: List[Tuple[str, str]],
                      metrics: List[str]) -> pd.DataFrame:
    """Compute Wilcoxon signed-rank tests for each (selectorA, selectorB) across k and metrics.
    Returns a tidy DataFrame and writes CSV under OUTDIR.
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy is required for Wilcoxon tests. Please pip install scipy.")

    rows = []
    ks = sorted(results.k.unique())
    for (sa, sb), k, metric in product(comparisons, ks, metrics):
        # Skip metric if not present
        if metric not in results.columns:
            continue
        a, b = _paired_vectors(results, sa, sb, k, metric)
        n = len(a)
        if n == 0:
            continue
        diff = a - b
        # If all diffs are exactly zero, wilcoxon fails; handle explicitly
        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0
        else:
            try:
                stat, p = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided', mode='auto')
            except ValueError:
                # fallback if ties/zeros weirdness
                stat, p = wilcoxon(a, b, zero_method='pratt', alternative='two-sided', mode='auto')
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        ci_lo = mean_diff - 1.96 * (std_diff / max(np.sqrt(n), 1.0)) if n > 1 else mean_diff
        ci_hi = mean_diff + 1.96 * (std_diff / max(np.sqrt(n), 1.0)) if n > 1 else mean_diff
        win_rate = float((diff > 0).mean())  # proportion where A > B
        rows.append({
            "selector_a": sa,
            "selector_b": sb,
            "k": k,
            "metric": metric,
            "n_pairs": n,
            "statistic": stat,
            "p_value": p,
            "mean_diff": mean_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "win_rate": win_rate,
        })
    sig = pd.DataFrame(rows)
    return sig


def plot_pvalues(sig: pd.DataFrame, outpng: str, alpha: float = 0.05):
    """Plot p-values vs k for each comparison and metric (one figure)."""
    if sig.empty:
        return
    plt.figure(figsize=(8, 5))
    # Generate a unique label per (selector_a, selector_b, metric)
    labels = sig.apply(lambda r: f"{r['selector_a']} vs {r['selector_b']} [{r['metric']}]", axis=1)
    sig = sig.assign(label=labels)
    for label, df in sig.groupby('label'):
        df_sorted = df.sort_values('k')
        plt.plot(df_sorted['k'], df_sorted['p_value'], marker='o', label=label)
    plt.axhline(alpha, linestyle='--', linewidth=1)
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('p-value (log scale)')
    plt.title('Wilcoxon p-values vs k')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()


def write_narrative(sig: pd.DataFrame, out_md: str, alpha: float = 0.05):
    """Auto-generate a short narrative summary from the significance table."""
    if sig.empty:
        open(out_md, 'w', encoding='utf-8').write("No significance results available.\n")
        return
    lines = ["# Statistical Significance Summary (Wilcoxon)", ""]
    for (sa, sb, metric), df in sig.groupby(["selector_a", "selector_b", "metric"]):
        df = df.sort_values("k")
        sig_ks = df.loc[df.p_value < alpha, "k"].tolist()
        trend = "significant" if sig_ks else "not significant"
        lines.append(f"**{sa} vs {sb}** on **{metric}**: {trend} at α={alpha}.")
        if sig_ks:
            lines.append(f"  - k where significant: {sig_ks}")
        # best k by mean_diff magnitude
        best_row = df.iloc[df["mean_diff"].abs().argmax()]
        md = best_row["mean_diff"]
        ci = (best_row["ci_lo"], best_row["ci_hi"])
        kbest = int(best_row["k"]) if not np.isnan(best_row["k"]) else None
        lines.append(f"  - Largest mean difference at k={kbest}: Δ={md:.4f} (95% CI {ci[0]:.4f}..{ci[1]:.4f}), p={best_row['p_value']:.3g}")
        lines.append("")
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


# ------------------------ ORIGINAL MAIN (extended, not replaced) ------------------------

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    results_path = os.path.join(OUTDIR, "results_cv.csv")
    if not os.path.exists(results_path):
        raise SystemExit(f"Missing {results_path}. Run your CV experiments to produce it.")

    results = pd.read_csv(results_path)

    # Curves vs k (averaged over classifiers to declutter)
    for metric in ["accuracy", "roc_auc"]:
        if metric not in results.columns:
            continue
        df = (
            results
            .groupby(["selector", "k", "fold_id"], as_index=False)[metric]
            .mean()
            .loc[:, ["selector", "k", metric]]
        )
        metric_vs_k(df, metric=metric, outpath=os.path.join(OUTDIR, f"fig_{metric}_vs_k.png"),
                    title=f"{metric.upper()} vs k (mean ± 95% CI)")

    # MI bar chart (top 10)
    mi_path = os.path.join(OUTDIR, "mi_scores.csv")
    if os.path.exists(mi_path):
        mi = pd.read_csv(mi_path).sort_values("mi", ascending=False)
        top_mi_bar(mi["mi"].values, mi["feature"].values, k=10, outpath=os.path.join(OUTDIR, "fig_top_mi.png"))

    # ROC curves at best-k for a fixed classifier (LogReg) across a few selectors
    X, y, feat_names, target_names = load_wdbc()
    summary_path = os.path.join(OUTDIR, "summary_by_method.csv")
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        clf_name = "LogReg"
        sels = ["MI", "mRMR", "FScore", "L1LogReg", "RFImp"]

        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=42,
                                                            stratify=y.values)

        plt.figure(figsize=(7, 5))
        roc_rows = []
        for sel_name in sels:
            row = summary[(summary.selector == sel_name) & (summary.classifier == clf_name)]
            if row.empty:
                continue
            best_k = int(row.iloc[0]["best_k"]) if not np.isnan(row.iloc[0]["best_k"]) else None
            if best_k is None:
                continue
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
            roc_rows.append({"fpr": fpr, "tpr": tpr, "label": f"{sel_name} (k={best_k})"})

        plot_roc_curves_bestk(roc_rows, os.path.join(OUTDIR, "fig_roc_bestk.png"))

    # ---------------- NEW: run Wilcoxon significance tests ----------------
    # Expect results_cv.csv to have at least: selector, k, fold_id, classifier, accuracy (optionally roc_auc, f1)
    comparisons = [("mRMR", "FScore"), ("MI", "FScore"), ("MI", "RFE")]
    metrics = [m for m in ["accuracy", "roc_auc", "f1"] if m in results.columns]
    if _HAVE_SCIPY and metrics:
        sig = run_wilcoxon_grid(results, comparisons, metrics)
        sig_csv = os.path.join(OUTDIR, "significance_wilcoxon.csv")
        sig.to_csv(sig_csv, index=False)
        plot_pvalues(sig, os.path.join(OUTDIR, "fig_wilcoxon_pvalues.png"))
        write_narrative(sig, os.path.join(OUTDIR, "significance_summary.md"))
    else:
        msg = []
        if not _HAVE_SCIPY:
            msg.append("scipy not installed;")
        if not metrics:
            msg.append("no metric columns among ['accuracy','roc_auc','f1'] present;")
        with open(os.path.join(OUTDIR, "significance_summary.md"), "w", encoding="utf-8") as f:
            f.write("Cannot run Wilcoxon tests: " + " ".join(msg) + "\n")

    print("Figures and significance artifacts written to ./outputs")


if __name__ == "__main__":
    main()
