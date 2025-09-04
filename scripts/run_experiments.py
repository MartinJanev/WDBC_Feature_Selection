import os, sys
# --- Path bootstrap so 'src.*' imports work regardless of CWD ---
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data import load_wdbc
from src.eval_protocol import evaluate
from src.selectors import MISelector

OUTDIR = os.path.join(PROJECT_ROOT, "outputs")

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    X, y, feat_names, target_names = load_wdbc()

    results, stability, summary = evaluate(
        X, y, k_grid=(3,5,8,10,15,20,30), repeats=5, folds=5, primary_metric="accuracy"
    )
    results.to_csv(os.path.join(OUTDIR, "results_cv.csv"), index=False)
    stability.to_csv(os.path.join(OUTDIR, "stability.csv"), index=False)
    summary.to_csv(os.path.join(OUTDIR, "summary_by_method.csv"), index=False)

    Xs = StandardScaler().fit_transform(X.values)
    mi_sel = MISelector(k=30)
    mi_sel.fit(Xs, y.values)
    mi_scores = mi_sel.scores_
    pd.DataFrame({"feature": feat_names, "mi": mi_scores}).to_csv(os.path.join(OUTDIR, "mi_scores.csv"), index=False)

    print("Done. See ./outputs. Next: python scripts/make_figures.py")

if __name__ == "__main__":
    main()
