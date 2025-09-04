import numpy as np
import pandas as pd
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
