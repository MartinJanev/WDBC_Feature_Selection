from __future__ import annotations
import itertools
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from .selectors import (
    MISelector, MRMRSelector, FScoreSelector, VarianceTopKSelector,
    L1LogRegSelector, RFImportanceSelector, RFELogRegSelector
)
from .utils import set_seed, mean_ci

@dataclass
class Config:
    selector_name: str
    selector_kwargs: dict
    classifier_name: str

def make_selector(name: str, k: int):
    if name == "MI": return MISelector(k=k)
    if name == "mRMR": return MRMRSelector(k=k, lam=1.0)
    if name == "FScore": return FScoreSelector(k=k)
    if name == "VarTopK": return VarianceTopKSelector(k=k)
    if name == "L1LogReg": return L1LogRegSelector(k=k, C=0.1)
    if name == "RFImp": return RFImportanceSelector(k=k)
    if name == "RFE-LogReg": return RFELogRegSelector(k=k)
    if name == "PCA":
        return None
    raise ValueError(name)

def make_classifier(name: str):
    if name == "LogReg":
        return LogisticRegression(max_iter=5000, random_state=42)
    if name == "LinearSVM":
        return SVC(kernel="linear", probability=True, random_state=42)
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    raise ValueError(name)

def evaluate(X, y, k_grid=(3,5,8,10,15,20,30), repeats=5, folds=5, primary_metric="accuracy"):
    set_seed(42)
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)

    selector_names = ["MI", "mRMR", "FScore", "VarTopK", "L1LogReg", "RFImp", "RFE-LogReg", "PCA"]
    classifier_names = ["LogReg", "LinearSVM", "RF"]

    rows = []
    stability_records = []
    fold_id = 0
    for train_idx, test_idx in tqdm(rskf.split(X, y), total=folds*repeats, desc="CV"):
        X_tr, X_te = X.iloc[train_idx].values, X.iloc[test_idx].values
        y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values

        for sel_name, clf_name in itertools.product(selector_names, classifier_names):
            for k in k_grid:
                selector = make_selector(sel_name, k)
                clf = make_classifier(clf_name)

                if sel_name == "PCA":
                    from sklearn.decomposition import PCA
                    pipe = Pipeline([
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=min(k, X.shape[1]))),
                        ("clf", clf),
                    ])
                else:
                    pipe = Pipeline([
                        ("selector", selector),
                        ("scaler", StandardScaler()),
                        ("clf", clf),
                    ])

                pipe.fit(X_tr, y_tr)
                y_pr = pipe.predict(X_te)

                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    y_proba = pipe.predict_proba(X_te)[:,1]
                else:
                    if hasattr(pipe.named_steps["clf"], "decision_function"):
                        scores = pipe.decision_function(X_te)
                        smin, smax = scores.min(), scores.max()
                        y_proba = (scores - smin) / (smax - smin + 1e-12)
                    else:
                        y_proba = None

                acc = accuracy_score(y_te, y_pr)
                f1  = f1_score(y_te, y_pr)
                roc_auc = roc_auc_score(y_te, y_proba) if y_proba is not None else np.nan

                rows.append(dict(
                    fold_id=fold_id, selector=sel_name, k=k, classifier=clf_name,
                    accuracy=acc, f1=f1, roc_auc=roc_auc
                ))

                if sel_name != "PCA":
                    sel = pipe.named_steps["selector"]
                    if hasattr(sel, "selected_indices_") and sel.selected_indices_ is not None:
                        stability_records.append(dict(
                            fold_id=fold_id, selector=sel_name, k=k,
                            selected=list(map(int, sel.selected_indices_))
                        ))

        fold_id += 1

    results = pd.DataFrame(rows)

    # Stability: average pairwise Jaccard across folds per (selector,k)
    stab_rows = []
    by_key = {}
    for r in stability_records:
        by_key.setdefault((r["selector"], r["k"]), []).append(set(r["selected"]))
    for (sel, k), sets in by_key.items():
        if len(sets) < 2:
            continue
        import itertools as it
        jaccards = []
        for a, b in it.combinations(sets, 2):
            inter = len(a & b)
            union = len(a | b)
            j = inter / union if union > 0 else 1.0
            jaccards.append(j)
        from .utils import mean_ci
        ci = mean_ci(jaccards)
        stab_rows.append(dict(selector=sel, k=k, jaccard_mean=ci.mean, jaccard_lo95=ci.lo95, jaccard_hi95=ci.hi95))
    stability = pd.DataFrame(stab_rows)

    metric = primary_metric
    grouped = results.groupby(["selector", "classifier", "k"], as_index=False)[metric].mean()
    best_rows = []
    for (sel, clf), g in grouped.groupby(["selector", "classifier"]):
        g = g.sort_values(metric, ascending=False)
        best_k = int(g.iloc[0]["k"])
        sub = results[(results["selector"]==sel) & (results["classifier"]==clf) & (results["k"]==best_k)]
        ci_acc = mean_ci(sub["accuracy"].values)
        ci_auc = mean_ci(sub["roc_auc"].values)
        ci_f1  = mean_ci(sub["f1"].values)
        best_rows.append(dict(
            selector=sel, classifier=clf, best_k=best_k,
            acc_mean=ci_acc.mean, acc_lo95=ci_acc.lo95, acc_hi95=ci_acc.hi95,
            auc_mean=ci_auc.mean, auc_lo95=ci_auc.lo95, auc_hi95=ci_auc.hi95,
            f1_mean=ci_f1.mean,   f1_lo95=ci_f1.lo95,   f1_hi95=ci_f1.hi95
        ))
    summary = pd.DataFrame(best_rows)

    return results, stability, summary
