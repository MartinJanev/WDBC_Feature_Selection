import os
import numpy as np
import matplotlib.pyplot as plt

def metric_vs_k(df, metric, outpath, title):
    plt.figure(figsize=(8,5))
    for (sel), gsel in df.groupby("selector"):
        stats = gsel.groupby("k")[metric].agg(["mean","std","count"]).reset_index()
        se = stats["std"] / np.sqrt(stats["count"])
        ci = 1.96 * se
        plt.errorbar(stats["k"], stats["mean"], yerr=ci, marker="o", capsize=3, label=sel)
    plt.xlabel("k (number of features / components)")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def top_mi_bar(mi_scores, feat_names, k, outpath):
    order = np.argsort(mi_scores)[::-1][:k]
    plt.figure(figsize=(8,5))
    y = np.array(feat_names)[order]
    x = mi_scores[order]
    pos = np.arange(len(order))
    plt.barh(pos, x)
    plt.yticks(pos, y)
    plt.xlabel("Mutual Information I(X;Y)")
    plt.title(f"Top {k} MI Scores")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_roc_curves_bestk(rows, out_png: str, title: str = "ROC curves @ best-k (LogReg)"):
    """
    rows: iterable of dicts with keys: {'fpr': np.ndarray, 'tpr': np.ndarray, 'label': str}
    """
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    for r in rows:
        fpr, tpr, label = r["fpr"], r["tpr"], r["label"]
        plt.plot(fpr, tpr, label=label)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=1, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def plot_pvalues(sig_df, out_png: str, alpha: float = 0.05):
    """
    Unified p-value plot for multiple (selector_a vs selector_b, metric) across k.
    Expects columns: ['selector_a','selector_b','metric','k','p_value']
    """
    if sig_df is None or len(sig_df) == 0:
        return
    plt.figure(figsize=(8, 5))
    labels = (sig_df["selector_a"] + " vs " + sig_df["selector_b"] + " [" + sig_df["metric"] + "]")
    for label, g in sig_df.assign(label=labels).groupby("label"):
        g = g.sort_values("k")
        plt.plot(g["k"], g["p_value"], marker="o", label=label)
    plt.axhline(alpha, linestyle="--", linewidth=1)
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("p-value (log scale)")
    plt.title("Wilcoxon p-values vs k")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_variability_bar(var_rows_df, out_png: str, metric: str = "accuracy"):
    """
    Grouped bars of across-fold variability per selector at small k.
    Expects columns: ['selector','k', f'{metric}_std'].
    """
    ks = sorted(var_rows_df["k"].unique())
    selectors = var_rows_df["selector"].unique().tolist()
    x = np.arange(len(selectors))
    width = 0.8 / max(1, len(ks))
    plt.figure(figsize=(10.5, 4.8))
    for i, k in enumerate(ks):
        g = var_rows_df[var_rows_df["k"] == k].set_index("selector")
        y = np.array([g.loc[s, f"{metric}_std"] if s in g.index else np.nan for s in selectors], float)
        plt.bar(x + i * width - (len(ks) - 1) * width / 2, y, width, label=f"k={k}")
    plt.xticks(x, selectors, rotation=30, ha="right")
    plt.ylabel(f"{metric.upper()} std across folds")
    plt.title("Across-fold variability at small k (classifier=LogReg)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_stability_bar(stab_summary_df, out_png: str, title="Selection stability at small k"):
    """
    Grouped bars with error bars for Jaccard stability.
    Expects ['selector','k','jaccard_mean','jaccard_lo95','jaccard_hi95'].
    """
    ks = sorted(stab_summary_df["k"].unique())
    selectors = stab_summary_df["selector"].unique().tolist()
    x = np.arange(len(selectors))
    width = 0.8 / max(1, len(ks))
    plt.figure(figsize=(10.5, 4.8))
    for i, k in enumerate(ks):
        g = stab_summary_df[stab_summary_df["k"] == k].set_index("selector")
        y  = np.array([g.loc[s, "jaccard_mean"] if s in g.index else np.nan for s in selectors], float)
        lo = np.array([g.loc[s, "jaccard_lo95"] if s in g.index else np.nan for s in selectors], float)
        hi = np.array([g.loc[s, "jaccard_hi95"] if s in g.index else np.nan for s in selectors], float)
        yerr = np.vstack([y - lo, hi - y])
        plt.bar(x + i * width - (len(ks) - 1) * width / 2, y, width, label=f"k={k}", yerr=yerr, capsize=3)
    plt.xticks(x, selectors, rotation=30, ha="right")
    plt.ylabel("Jaccard stability (mean ± 95% CI)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()