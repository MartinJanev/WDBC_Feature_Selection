# selectors_mi.py
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from src.selectors_standard import BaseSelector

__all__ = ["MISelector", "MRMRSelector"]


class MISelector(BaseSelector):
    """
    Mutual Information (MI) feature selector.
    Uses sklearn's mutual_info_classif to compute MI between each feature and the target.
    """

    def __init__(self, k: int, n_neighbors: int = 3, random_state: int = 42):
        super().__init__(k)
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._scaler = StandardScaler()

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        mi = mutual_info_classif(
            Xs, y, n_neighbors=self.n_neighbors, random_state=self.random_state
        )
        return self._finalize_support(mi)


class MRMRSelector(BaseSelector):
    """
    Minimum Redundancy Maximum Relevance (mRMR) feature selector.
    """

    def __init__(self, k: int, lam: float = 1.0, n_neighbors: int = 3, random_state: int = 42):
        super().__init__(k)
        self.lam = lam
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._scaler = StandardScaler()

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        n_features = Xs.shape[1]
        rel = mutual_info_classif(
            Xs, y, n_neighbors=self.n_neighbors, random_state=self.random_state
        )

        selected = []
        remaining = list(range(n_features))
        scores = np.zeros(n_features)
        pair_cache: dict[tuple[int, int], float] = {}

        def mi_feat(a: int, b: int) -> float:
            key = (min(a, b), max(a, b))
            if key in pair_cache:
                return pair_cache[key]
            val = mutual_info_regression(
                Xs[:, [a]], Xs[:, b], n_neighbors=self.n_neighbors, random_state=self.random_state
            )[0]
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
