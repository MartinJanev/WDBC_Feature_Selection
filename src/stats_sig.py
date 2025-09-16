# src/stats_sig.py
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def paired_vectors(df: pd.DataFrame, sel_a: str, sel_b: str, k: int, metric: str):
    cols = ["fold_id", "classifier"]
    da = (df[(df.selector == sel_a) & (df.k == k)]
          .loc[:, [*cols, metric]]
          .rename(columns={metric: f"{metric}_a"}))
    db = (df[(df.selector == sel_b) & (df.k == k)]
          .loc[:, [*cols, metric]]
          .rename(columns={metric: f"{metric}_b"}))
    merged = pd.merge(da, db, on=cols, how="inner")
    return merged[f"{metric}_a"].to_numpy(), merged[f"{metric}_b"].to_numpy()

def run_wilcoxon_grid(results: pd.DataFrame, comparisons, metrics):
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy is required for Wilcoxon tests. Please pip install scipy.")
    rows = []
    ks = sorted(results.k.unique())
    for (sa, sb) in comparisons:
        for k in ks:
            for metric in metrics:
                if metric not in results.columns:
                    continue
                a, b = paired_vectors(results, sa, sb, k, metric)
                n = len(a)
                if n == 0:
                    continue
                diff = a - b
                if np.allclose(diff, 0):
                    stat, p = np.nan, 1.0
                else:
                    try:
                        stat, p = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided', mode='auto')
                    except ValueError:
                        stat, p = wilcoxon(a, b, zero_method='pratt', alternative='two-sided', mode='auto')
                mean_diff = float(np.mean(diff))
                std_diff  = float(np.std(diff, ddof=1)) if n > 1 else 0.0
                ci_lo = mean_diff - 1.96 * (std_diff / max(np.sqrt(n), 1.0)) if n > 1 else mean_diff
                ci_hi = mean_diff + 1.96 * (std_diff / max(np.sqrt(n), 1.0)) if n > 1 else mean_diff
                win_rate = float((diff > 0).mean())
                rows.append(dict(selector_a=sa, selector_b=sb, k=k, metric=metric,
                                 n_pairs=n, statistic=stat, p_value=p,
                                 mean_diff=mean_diff, ci_lo=ci_lo, ci_hi=ci_hi, win_rate=win_rate))
    return pd.DataFrame(rows)

def write_narrative(sig: pd.DataFrame, out_md: str, alpha: float = 0.05):
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
        best_row = df.iloc[df["mean_diff"].abs().argmax()]
        md = best_row["mean_diff"]; kbest = int(best_row["k"])
        ci = (best_row["ci_lo"], best_row["ci_hi"])
        lines.append(f"  - Largest mean difference at k={kbest}: Δ={md:.4f} (95% CI {ci[0]:.4f}..{ci[1]:.4f}), p={best_row['p_value']:.3g}")
        lines.append("")
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
