# scripts/MI_vs_Standard.py
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root on path (so running from anywhere works)
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUTDIR = os.path.join(PROJECT_ROOT, "outputs")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def ci_lo(x):
    n = x.count()
    if n <= 1:
        return x.mean()
    return x.mean() - 1.96 * (x.std(ddof=1) / math.sqrt(n))


def ci_hi(x):
    n = x.count()
    if n <= 1:
        return x.mean()
    return x.mean() + 1.96 * (x.std(ddof=1) / math.sqrt(n))


def _plot_family_metric(summary, metric, out_png, title):
    """
    Draw mean +/- 95% CI vs k for MI vs Standard families.
    Expects 'summary' to have columns: family, k, {metric}_mean, {metric}_lo95, {metric}_hi95
    """
    fams = ["MI", "Standard"]
    plt.figure(figsize=(7.5, 4.6))
    for fam in fams:
        g = summary[summary["family"] == fam].sort_values("k")
        if g.empty:
            continue
        y = g[f"{metric}_mean"].values
        lo = g[f"{metric}_lo95"].values
        hi = g[f"{metric}_hi95"].values
        x = g["k"].values
        # asymmetric yerr
        yerr = np.vstack([y - lo, hi - y])
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=fam)
    plt.xlabel("k (number of features / components)")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_bestk_table(best, out_png, title):
    # Bar chart per family showing acc_mean at best-k; annotate with best_k
    plt.figure(figsize=(6.2, 4))
    x = np.arange(len(best))
    y = best["acc_mean"].values
    labs = best["family"].values
    bars = plt.bar(x, y)
    for i, b in enumerate(bars):
        bk = int(best.iloc[i]["best_k"])
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1e-3, f"best k={bk}", ha="center", va="bottom",
                 fontsize=9)
    plt.xticks(x, labs)
    plt.ylabel("Accuracy (mean at best-k)")
    plt.title(title)
    plt.ylim(0, max(1.0, y.max() + 0.02))
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    
    # The objective here is to summarize results by feature selector family (MI-based vs Standard)
    # and to visualize family-level performance vs k (number of features/components selected).
    # We treat folds × classifiers as replicates when aggregating by family & k.
    
    
    csv_path = os.path.join(OUTDIR, "results_cv.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"Missing {csv_path}. Run: python scripts/run_experiments.py")

    df = pd.read_csv(csv_path)

    # Map selectors to families
    family = {
        "MI": "MI", "mRMR": "MI",
        "FScore": "Standard", "VarTopK": "Standard",
        "L1LogReg": "Standard", "RFImp": "Standard",
        "RFE-LogReg": "Standard", "PCA": "Standard",
    }
    df["family"] = df["selector"].map(family)

    # Aggregate by family & k (treating folds × classifiers as replicates)
    summary = (
        df.groupby(["family", "k"])
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_lo95=("accuracy", ci_lo),
            acc_hi95=("accuracy", ci_hi),
            auc_mean=("roc_auc", "mean"),
            auc_lo95=("roc_auc", ci_lo),
            auc_hi95=("roc_auc", ci_hi),
            f1_mean=("f1", "mean"),
            f1_lo95=("f1", ci_lo),
            f1_hi95=("f1", ci_hi),
            n=("accuracy", "count"),
        )
        .reset_index()
        .sort_values(["family", "k"])
    )

    # Best-k per family (by accuracy mean)
    best = (
        summary.sort_values(["family", "acc_mean"], ascending=[True, False])
        .groupby("family", as_index=False).first()
        .loc[:, ["family", "k", "acc_mean", "auc_mean", "f1_mean"]]
        .rename(columns={"k": "best_k"})
    )

    out_csv = os.path.join(OUTDIR, "summary_by_family.csv")
    summary.to_csv(out_csv, index=False)

    # ---- Plots ----
    _plot_family_metric(summary, "acc",
                        os.path.join(OUTDIR, "fig_family_accuracy_vs_k.png"),
                        "Family-level Accuracy vs k (mean ± 95% CI)")

    _plot_family_metric(summary, "auc",
                        os.path.join(OUTDIR, "fig_family_rocauc_vs_k.png"),
                        "Family-level ROC-AUC vs k (mean ± 95% CI)")

    _plot_family_metric(summary, "f1",
                        os.path.join(OUTDIR, "fig_family_f1_vs_k.png"),
                        "Family-level F1 vs k (mean ± 95% CI)")

    _plot_bestk_table(best,
                      os.path.join(OUTDIR, "fig_family_bestk_acc.png"),
                      "Accuracy at family best-k (annotated with best k)")

    print("Wrote:",
          out_csv, ",",
          "fig_family_accuracy_vs_k.png, fig_family_rocauc_vs_k.png, fig_family_f1_vs_k.png, fig_family_bestk_acc.png",
          "to outputs/.")


if __name__ == "__main__":
    main()
