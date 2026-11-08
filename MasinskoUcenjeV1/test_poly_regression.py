"""Zadatak 2: ucitavanje podataka, obuka, predikcija i MSE."""
from pathlib import Path

import pandas as pd
from poly_regression import PolynomialRegression


def main():
    path = Path(__file__).resolve().parent / "podaci.csv"
    table = pd.read_csv(path)
    X = table[["X"]].to_numpy()
    y = table["Y"].to_numpy()

    model = PolynomialRegression(degree=2)
    model.fit(X, y)
    predictions = model.predict(X)

    print("Model: polinom 2. stepena (M=2)")
    print("Parametri (w0, w1, w2):", model.coef_)
    print("Predikcije:")
    for x, true, predicted in zip(X[:, 0], y, predictions):
        print(f"X={x:.0f} mm | izmereno={true:.4f} | predvidjeno={predicted:.4f}")
    print(f"MSE: {model.score(X, y):.9f}")
    print(f"Err = MSE/2: {model.score(X, y) / 2:.9f}")
    print(f"Predikcija za 900 mm: {model.predict([[900]])[0]:.6f} t/ha")


if __name__ == "__main__":
    main()
