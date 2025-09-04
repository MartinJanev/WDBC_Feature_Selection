from __future__ import annotations
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.utils.validation import check_is_fitted

class BaseSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k: int):
        self.k = k
        self.support_ = None
        self.selected_indices_: Optional[np.ndarray] = None

    def get_support(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_

    def _finalize_support(self, scores: np.ndarray):
        k = min(self.k, len(scores))
        idx = np.argsort(scores)[::-1][:k]
        mask = np.zeros(len(scores), dtype=bool)
        mask[idx] = True
        self.support_ = mask
        self.selected_indices_ = idx
        self.scores_ = scores
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        return X[:, self.support_]

class MISelector(BaseSelector):
    def __init__(self, k: int, n_neighbors: int = 3, random_state: int = 42):
        super().__init__(k)
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._scaler = StandardScaler()

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        mi = mutual_info_classif(Xs, y, n_neighbors=self.n_neighbors, random_state=self.random_state)
        return self._finalize_support(mi)

class MRMRSelector(BaseSelector):
    def __init__(self, k: int, lam: float = 1.0, n_neighbors: int = 3, random_state: int = 42):
        super().__init__(k)
        self.lam = lam
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._scaler = StandardScaler()

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        n_features = Xs.shape[1]
        rel = mutual_info_classif(Xs, y, n_neighbors=self.n_neighbors, random_state=self.random_state)

        selected = []
        remaining = list(range(n_features))
        scores = np.zeros(n_features)
        pair_cache = {}

        def mi_feat(a, b):
            key = (min(a,b), max(a,b))
            if key in pair_cache:
                return pair_cache[key]
            val = mutual_info_regression(Xs[:, [a]], Xs[:, b], n_neighbors=self.n_neighbors, random_state=self.random_state)[0]
            pair_cache[key] = val
            return val

        for _ in range(min(self.k, n_features)):
            best_j, best_val = None, -np.inf
            for j in remaining:
                if not selected:
                    val = rel[j]
                else:
                    red = np.mean([mi_feat(j, s) for s in selected])
                    val = rel[j] - self.lam * red
                if val > best_val:
                    best_val = val
                    best_j = j
                scores[j] = val
            selected.append(best_j)
            remaining.remove(best_j)

        mask = np.zeros(n_features, dtype=bool)
        mask[selected] = True
        self.support_ = mask
        self.selected_indices_ = np.array(selected, dtype=int)
        self.scores_ = scores
        self.relevance_ = rel
        return self

class FScoreSelector(BaseSelector):
    def __init__(self, k: int):
        super().__init__(k)

    def fit(self, X, y):
        f_vals, _ = f_classif(X, y)
        f_vals = np.nan_to_num(f_vals, nan=-np.inf)
        return self._finalize_support(f_vals)

class VarianceTopKSelector(BaseSelector):
    def __init__(self, k: int):
        super().__init__(k)

    def fit(self, X, y=None):
        variances = np.var(X, axis=0, ddof=1)
        return self._finalize_support(variances)

class L1LogRegSelector(BaseSelector):
    def __init__(self, k: int, C: float = 0.1, max_iter: int = 5000, random_state: int = 42):
        super().__init__(k)
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(
            penalty="l1", solver="saga", C=self.C, max_iter=self.max_iter, random_state=self.random_state
        )

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        self._clf.fit(Xs, y)
        coefs = np.abs(self._clf.coef_).ravel()
        return self._finalize_support(coefs)

class RFImportanceSelector(BaseSelector):
    def __init__(self, k: int, n_estimators: int = 500, random_state: int = 42):
        super().__init__(k)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._rf = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1
        )

    def fit(self, X, y):
        self._rf.fit(X, y)
        importances = self._rf.feature_importances_
        return self._finalize_support(importances)

class RFELogRegSelector(BaseSelector):
    def __init__(self, k: int, max_iter: int = 5000, random_state: int = 42):
        super().__init__(k)
        self._scaler = StandardScaler()
        self._est = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self._rfe = RFE(self._est, n_features_to_select=k, step=1)

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        self._rfe.fit(Xs, y)
        self.support_ = self._rfe.support_
        self.selected_indices_ = np.where(self.support_)[0]
        self.scores_ = (1.0 / (self._rfe.ranking_.astype(float)))
        return self
