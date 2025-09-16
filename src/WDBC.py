from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_wdbc():
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    target_names = list(ds.target_names)
    return X, y, ds.feature_names, target_names
