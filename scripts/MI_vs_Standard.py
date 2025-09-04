# scripts/MI_vs_Standard.py
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel

# --- Path bootstrap so 'src.*' is importable if needed ---
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUTDIR = os.path.join(PROJECT_ROOT, "outputs")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MI_METHODS = ["MI", "mRMR"]
STD_METHODS = ["FScore", "VarTopK", "L1LogReg", "RFImp", "RFE-LogReg", "PCA"]


def load_results():
    rcsv = os.path.join(OUTDIR, "results_cv.csv")
    scsv = os.path.join(OUTDIR, "stability.csv")
    if not os.path.exists(rcsv):
        raise SystemExit("Missing outputs/results_cv.csv. Run: python scripts/run_experiments.py")
    res = pd.read_csv(rcsv)
    stab = pd.read_csv(scsv) if os.path.exists(scsv) else None
    return res, stab


def k_at_threshold(df, metric="accuracy", thresh=0.95):
    """For each selector: min k where mean(metric|k) ≥ thresh * max_k mean(metric|k)."""
    rows = []
    for sel, g in df.groupby("selector"):
        means = g.groupby("k")[metric].mean().sort_index()
        peak = means.max()
        cutoff = thresh * peak
        ks = means.index.values
        vals = means.values
        try:
            eff_k = int(ks[np.where(vals >= cutoff)[0][0]])
        except IndexError:
            eff_k = int(ks[-1])
        rows.append(dict(selector=sel, peak_mean=float(peak), k_at_95=eff_k))
    return pd.DataFrame(rows).sort_values("k_at_95")


def paired_compare(res, sel_a, sel_b, k, metric="accuracy", clf="LogReg"):
    """Fold-wise paired test between two selectors at fixed k & classifier."""
    a = res[(res.selector == sel_a) & (res.k == k) & (res.classifier == clf)][["fold_id", metric]].set_index("fold_id")
    b = res[(res.selector == sel_b) & (res.k == k) & (res.classifier == clf)][["fold_id", metric]].set_index("fold_id")
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return None
    da = a.loc[idx, metric].values
    db = b.loc[idx, metric].values
    diff = da - db
    dbar = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    cohen_d = (dbar / sd) if sd > 0 else np.nan
    # Include zeros via 'pratt' (safer when many ties)
    w = wilcoxon(diff, zero_method="pratt", alternative="two-sided")
    t = ttest_rel(da, db)
    return dict(selector_A=sel_a, selector_B=sel_b, k=int(k), classifier=clf,
                n=len(diff), mean_diff=dbar, cohen_d=cohen_d,
                p_wilcoxon=float(w.pvalue), p_ttest=float(t.pvalue))


def _plot_k_efficiency(eff_df, out_png):
    eff_df = eff_df.copy().reset_index(drop=True)
    # Color by family: MI vs Standard
    fam = eff_df["selector"].map(lambda s: "MI" if s in MI_METHODS else "Standard")
    colors = ["#1f77b4" if f == "MI" else "#ff7f0e" for f in fam]
    order = eff_df.sort_values("k_at_95").index.values
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(np.arange(len(order)), eff_df.loc[order, "k_at_95"].values, color=[colors[i] for i in order])
    plt.xticks(np.arange(len(order)), eff_df.loc[order, "selector"].values, rotation=30, ha="right")
    plt.ylabel("Min k to reach 95% of own peak (Accuracy)")
    plt.title("Feature-efficiency: k@95% peak per selector (classifier=LogReg)")
    plt.grid(True, axis="y", alpha=0.25)
    # Legend manually
    mi_patch = plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="MI family")
    std_patch = plt.Rectangle((0, 0), 1, 1, color="#ff7f0e", label="Standard")
    plt.legend(handles=[mi_patch, std_patch], loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_paired_tests_bars(tests, k, out_png):
    dfk = tests[tests["k"] == k].copy()
    if dfk.empty:
        return
    # Build grouped bars: for each baseline selector_B, bars for selector_A in ["MI","mRMR"]
    bases = [b for b in STD_METHODS if b in dfk["selector_B"].unique()]
    x = np.arange(len(bases))
    width = 0.38

    def grab(selA):
        return [dfk[(dfk.selector_B == b) & (dfk.selector_A == selA)]["mean_diff"].mean() for b in bases]

    y_mi = grab("MI")
    y_mrmr = grab("mRMR")

    plt.figure(figsize=(9, 4.6))
    plt.axhline(0, color="k", linewidth=0.8, alpha=0.7)
    b1 = plt.bar(x - width / 2, y_mi, width, label="MI − baseline")
    b2 = plt.bar(x + width / 2, y_mrmr, width, label="mRMR − baseline")
    plt.xticks(x, bases, rotation=30, ha="right")
    plt.ylabel("Mean fold-wise ACC difference")
    plt.title(f"Paired fold-wise differences at k={k} (classifier=LogReg)\n(positive = MI/mRMR higher)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    # Optionally annotate p-values above bars (Wilcoxon), if available
    for i, b in enumerate(b1):
        base = bases[i]
        row = dfk[(dfk.selector_B == base) & (dfk.selector_A == "MI")]
        if not row.empty:
            p = row["p_wilcoxon"].values[0]
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1e-3, f"p={p:.3f}", ha="center", va="bottom",
                     fontsize=8)
    for i, b in enumerate(b2):
        base = bases[i]
        row = dfk[(dfk.selector_B == base) & (dfk.selector_A == "mRMR")]
        if not row.empty:
            p = row["p_wilcoxon"].values[0]
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1e-3, f"p={p:.3f}", ha="center", va="bottom",
                     fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_variability(var_df, metric, out_png):
    # Grouped bars by k; x-axis selectors
    ks = sorted(var_df["k"].unique())
    selectors = var_df["selector"].unique().tolist()
    x = np.arange(len(selectors))
    width = 0.8 / max(1, len(ks))
    plt.figure(figsize=(10, 4.6))
    for i, k in enumerate(ks):
        g = var_df[var_df["k"] == k].set_index("selector")
        y = [g.loc[s, f"{metric}_std"] if s in g.index else 0.0 for s in selectors]
        plt.bar(x + i * width - (len(ks) - 1) * width / 2, y, width, label=f"k={k}")
    plt.xticks(x, selectors, rotation=30, ha="right")
    plt.ylabel(f"{metric.upper()} std across folds")
    plt.title(f"Across-fold variability at small k (classifier=LogReg)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_stability(stab_summary, out_png):
    # Grouped bars by k for each selector, with error bars from lo/hi
    ks = sorted(stab_summary["k"].unique())
    selectors = stab_summary["selector"].unique().tolist()
    x = np.arange(len(selectors))
    width = 0.8 / max(1, len(ks))
    plt.figure(figsize=(10.5, 4.8))
    for i, k in enumerate(ks):
        g = stab_summary[stab_summary["k"] == k].set_index("selector")
        y = [g.loc[s, "jaccard_mean"] if s in g.index else np.nan for s in selectors]
        lo = [g.loc[s, "jaccard_lo95"] if s in g.index else np.nan for s in selectors]
        hi = [g.loc[s, "jaccard_hi95"] if s in g.index else np.nan for s in selectors]
        y = np.array(y, float)
        lo = np.array(lo, float);
        hi = np.array(hi, float)
        yerr = np.vstack([y - lo, hi - y])
        plt.bar(x + i * width - (len(ks) - 1) * width / 2, y, width, label=f"k={k}", yerr=yerr, capsize=3)
    plt.xticks(x, selectors, rotation=30, ha="right")
    plt.ylabel("Jaccard stability (mean ± 95% CI)")
    plt.title("Selection stability at small k")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():

    # The objective here is to compare MI-based methods (MI, mRMR) against standard methods
    # using the results in outputs/results_cv.csv and outputs/stability.csv (if present).
    # We focus on small k (5, 8, 10) where differences are
    # more likely to appear, and on a fixed classifier (LogReg) to avoid mixing effects.
    # We produce:
    # - Feature-efficiency: k@95% of own peak accuracy per selector
    # - Paired tests: MI & mRMR vs each standard method at k=
    # - Across-fold variability at small k
    # - Jaccard stability at small k (if stability.csv is present)


    res, stab = load_results()

    # Focus comparisons on a fixed classifier to avoid mixing effects
    clf = "LogReg"
    res_clf = res[res["classifier"] == clf].copy()

    # Feature-efficiency
    eff = k_at_threshold(res_clf, metric="accuracy", thresh=0.95)
    eff.to_csv(os.path.join(OUTDIR, "k_efficiency_95.csv"), index=False)
    _plot_k_efficiency(eff, os.path.join(OUTDIR, "fig_k_efficiency_95.png"))

    # Paired tests: MI & mRMR vs each standard method at small k
    ks = [5, 8, 10]
    rows = []
    for k in ks:
        for base in STD_METHODS:
            for mi in MI_METHODS:
                r = paired_compare(res_clf, mi, base, k, metric="accuracy", clf=clf)
                if r: rows.append(r)
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["wins_A"] = tests["mean_diff"] > 0
        tests.to_csv(os.path.join(OUTDIR, "paired_tests_smallk.csv"), index=False)
        # Plots per k
        for k in ks:
            _plot_paired_tests_bars(tests, k, os.path.join(OUTDIR, f"fig_paired_diffs_k{k}.png"))

    # Across-fold variability at small k
    var_rows = []
    for sel, k in [(s, k) for s in (MI_METHODS + STD_METHODS) for k in ks]:
        g = res_clf[(res_clf.selector == sel) & (res_clf.k == k)]
        if g.empty: continue
        var_rows.append(dict(selector=sel, k=k,
                             acc_mean=g["accuracy"].mean(),
                             acc_std=g["accuracy"].std(ddof=1),
                             auc_mean=g["roc_auc"].mean(),
                             auc_std=g["roc_auc"].std(ddof=1)))
    var_df = pd.DataFrame(var_rows).sort_values(["k", "selector"])
    if not var_df.empty:
        var_df.to_csv(os.path.join(OUTDIR, "metric_variability_smallk.csv"), index=False)
        _plot_variability(var_df, "acc", os.path.join(OUTDIR, "fig_variability_acc_smallk.png"))
        _plot_variability(var_df, "auc", os.path.join(OUTDIR, "fig_variability_auc_smallk.png"))

    # Jaccard stability
    if stab is not None and not stab.empty:
        smallk = stab[stab["k"].isin(ks)].copy()
        smallk["family"] = np.where(smallk["selector"].isin(MI_METHODS), "MI", "Standard")
        stab_summary = (smallk.groupby(["selector", "k"], as_index=False)
                        .agg(jaccard_mean=("jaccard_mean", "mean"),
                             jaccard_lo95=("jaccard_lo95", "mean"),
                             jaccard_hi95=("jaccard_hi95", "mean")))
        stab_summary.to_csv(os.path.join(OUTDIR, "stability_smallk.csv"), index=False)
        _plot_stability(stab_summary, os.path.join(OUTDIR, "fig_stability_smallk.png"))

    print("Wrote plots to outputs/: fig_k_efficiency_95.png, fig_paired_diffs_k{5,8,10}.png, "
          "fig_variability_acc_smallk.png, fig_variability_auc_smallk.png, fig_stability_smallk.png")


if __name__ == "__main__":
    main()
