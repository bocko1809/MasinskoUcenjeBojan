"""Polinomijalna regresija sa jednom ulaznom promenljivom.

Model: y(x) = w0 + w1*x + ... + wM*x**M.
Funkcija score namerno vraca MSE, prema formulaciji vezbe.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class PolynomialRegression(RegressorMixin, BaseEstimator):
    """Regresija normalnim jednacinama, uz opcionu L2 regularizaciju.

    degree: stepen polinoma M.
    alpha: koeficijent u Err = SSE/(2N) + alpha*sum(w_j**2, j>=1).
    solver: 'numpy', 'high_precision' ili 'auto'. Kod 'auto' koristi se
            veca preciznost za M >= 7, jer je Vandermondeova matrica
            za sirove vrednosti u mm vrlo lose uslovljena.
    decimal_digits: broj decimalnih cifara pri racunanju visokom preciznoscu.

    Ulazna promenljiva X se NE normalizuje i NE standardizuje.
    """

    def __init__(self, degree=2, alpha=0.0, solver="auto", decimal_digits=100):
        self.degree = degree
        self.alpha = alpha
        self.solver = solver
        self.decimal_digits = decimal_digits

    @staticmethod
    def _x_vector(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2 and X.shape[1] == 1:
            X = X[:, 0]
        elif X.ndim != 1:
            raise ValueError("X mora da ima jednu kolonu.")
        if X.size == 0 or not np.all(np.isfinite(X)):
            raise ValueError("X mora sadrzati konacne numericke vrednosti.")
        return X

    def _design_matrix(self, X):
        """Phi[i, j] = X[i]**j, bez transformacije ulaznih vrednosti."""
        return np.vander(X, N=self.degree + 1, increasing=True)

    def fit(self, X, y):
        if isinstance(self.degree, bool) or not isinstance(self.degree, (int, np.integer)) or self.degree < 0:
            raise ValueError("degree mora biti nenegativan ceo broj.")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha mora biti nenegativan konacan broj.")
        if self.solver not in ("numpy", "high_precision", "auto"):
            raise ValueError("solver mora biti 'numpy', 'high_precision' ili 'auto'.")
        if not isinstance(self.decimal_digits, int) or self.decimal_digits < 30:
            raise ValueError("decimal_digits mora biti ceo broj >= 30.")

        x = self._x_vector(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(x) != len(y) or not np.all(np.isfinite(y)):
            raise ValueError("X i y moraju imati jednak broj konacnih vrednosti.")
        self.n_features_in_ = 1
        self.n_samples_ = len(x)
        self._mp_coefficients = None
        chosen = ("high_precision" if self.degree >= 7 else "numpy") if self.solver == "auto" else self.solver
        self.solver_used_ = chosen

        if chosen == "numpy":
            phi = self._design_matrix(x)
            a = phi.T @ phi
            b = phi.T @ y
            # Izvod SSE/(2N) + alpha*sum(w_j^2): faktor 2*N*alpha.
            penalty = np.diag([0.0] + [1.0] * self.degree)
            a = a + (2 * len(x) * self.alpha) * penalty
            self.coef_ = np.linalg.solve(a, b)
        else:
            # Ista formula normalnih jednacina, ali sa vise cifara.
            # Bez vece preciznosti M=9 daje nepouzdane koeficijente.
            import mpmath as mp

            with mp.workdps(self.decimal_digits):
                xx = [mp.mpf(str(value)) for value in x]
                yy = mp.matrix([mp.mpf(str(value)) for value in y])
                phi = mp.matrix([[value**j for j in range(self.degree + 1)] for value in xx])
                a = phi.T * phi
                b = phi.T * yy
                for j in range(1, self.degree + 1):
                    a[j, j] += 2 * len(x) * mp.mpf(str(self.alpha))
                w = mp.lu_solve(a, b)
                self._mp_coefficients = tuple(mp.mpf(w[j]) for j in range(self.degree + 1))
                self.coef_ = np.array([float(value) for value in self._mp_coefficients])
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        x = self._x_vector(X)
        if self._mp_coefficients is None:
            return self._design_matrix(x) @ self.coef_

        import mpmath as mp
        with mp.workdps(self.decimal_digits):
            result = []
            for value in x:
                v = mp.mpf(str(value))
                # Hornerov algoritam: vrednovanje polinoma bez gubitka preciznosti.
                p = mp.mpf("0")
                for coefficient in reversed(self._mp_coefficients):
                    p = p * v + coefficient
                result.append(float(p))
        return np.array(result, dtype=float)

    def score(self, X, y):
        """Srednja kvadratna greska (MSE); manje je bolje."""
        y = np.asarray(y, dtype=float).reshape(-1)
        prediction = self.predict(X)
        if len(y) != len(prediction):
            raise ValueError("X i y moraju imati isti broj merenja.")
        return float(np.mean((y - prediction) ** 2))

    def objective(self, X, y):
        """Vrednost funkcije cilja, ukljucujuci L2 kaznu kada je alpha > 0."""
        mse = self.score(X, y)
        return 0.5 * mse + self.alpha * float(np.sum(self.coef_[1:] ** 2))
