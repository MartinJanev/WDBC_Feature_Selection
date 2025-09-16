# selectors_standard.py
from __future__ import annotations
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "BaseSelector",
    "FScoreSelector",
    "VarianceTopKSelector",
    "L1LogRegSelector",
    "RFImportanceSelector",
    "RecFeatElimLogRegSelector",
]

class BaseSelector(BaseEstimator, TransformerMixin):
    """
    Base class for feature selectors.
    Implements common functionality for all selectors.
    :param k: Number of top features to select
    """
    def __init__(self, k: int):
        self.k = k
        self.support_: Optional[np.ndarray] = None
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

class FScoreSelector(BaseSelector):
    """
    Feature selector based on ANOVA F-values.
    Uses sklearn's f_classif to compute F-values between each feature and the target.
    """
    def __init__(self, k: int):
        super().__init__(k)

    def fit(self, X, y):
        f_vals, _ = f_classif(X, y)
        f_vals = np.nan_to_num(f_vals, nan=-np.inf)
        return self._finalize_support(f_vals)

class VarianceTopKSelector(BaseSelector):
    """
    Feature selector that selects features based on their variance.
    """
    def __init__(self, k: int):
        super().__init__(k)

    def fit(self, X, y=None):
        variances = np.var(X, axis=0, ddof=1)
        return self._finalize_support(variances)

class L1LogRegSelector(BaseSelector):
    """
    Feature selector using L1-regularized Logistic Regression.
    """
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
    """
    Feature selector based on feature importances from a Random Forest classifier.
    """
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

class RecFeatElimLogRegSelector(BaseSelector):
    """
    Recursive Feature Elimination (RFE) with Logistic Regression as the estimator.
    """
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
