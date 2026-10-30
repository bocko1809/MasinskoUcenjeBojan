"""Kratke provere svih delova vezbe."""
from pathlib import Path

import numpy as np
import pandas as pd
from poly_regression import PolynomialRegression

ROOT = Path(__file__).resolve().parent
frame = pd.read_csv(ROOT / "podaci.csv")
X, y = frame[["X"]].to_numpy(), frame["Y"].to_numpy()

model2 = PolynomialRegression(degree=2).fit(X, y)
assert np.isclose(model2.score(X, y), 0.05755136383219559, rtol=1e-8)
assert np.isclose(model2.predict([[900]])[0], 5.54972229953, rtol=1e-8)

for degree in [1, 2, 3, 5, 9]:
    model = PolynomialRegression(degree=degree).fit(X, y)
    assert len(model.coef_) == degree + 1
    assert np.all(np.isfinite(model.predict([[600], [900], [1300], [1600]])))
    assert np.isclose(model.objective(X, y), model.score(X, y) / 2, rtol=1e-6, atol=1e-10)

model9 = PolynomialRegression(degree=9).fit(X, y)
assert model9.score(X, y) < 1e-18
assert model9.predict([[1600]])[0] > 50

for degree in [2, 5, 9]:
    ridge = PolynomialRegression(degree=degree, alpha=100).fit(X, y)
    base = PolynomialRegression(degree=degree, alpha=0).fit(X, y)
    assert ridge.objective(X, y) <= (base.score(X, y) / 2 + 100 * np.sum(base.coef_[1:] ** 2)) + 1e-5

print("PROVERA USPESNA: zadaci 2, 3 i 4 rade za zadate podatke.")
