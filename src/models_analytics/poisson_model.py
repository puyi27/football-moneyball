from typing import Tuple, Dict, List
import numpy as np
from scipy.stats import poisson


class DixonColesModel:
    """
    Motor matemático puro del modelo de Poisson Bivariado modificado de Dixon y Coles (1997).
    Ajusta la dependencia estocástica entre los goles del local y visitante para marcadores
    bajos (0-0, 1-0, 0-1, 1-1), eliminando la subestimación de empates típica de Poisson puro.
    Totalmente desacoplado de bases de datos y frameworks web.
    """

    def __init__(self, max_goals: int = 6):
        self.max_goals = max(3, int(max_goals))

    def tau_adjustment(self, x: int, y: int, lambda_: float, mu: float, rho: float) -> float:
        """
        Calcula el factor de corrección tau(x, y; lambda, mu, rho) de Dixon y Coles.
        Garantiza no-negatividad estricta para preservar los axiomas de Kolmogorov:
        - (0, 0): 1 - lambda * mu * rho
        - (0, 1): 1 + lambda * rho
        - (1, 0): 1 + mu * rho
        - (1, 1): 1 - rho
        - resto: 1.0
        """
        if x == 0 and y == 0:
            val = 1.0 - lambda_ * mu * rho
        elif x == 0 and y == 1:
            val = 1.0 + lambda_ * rho
        elif x == 1 and y == 0:
            val = 1.0 + mu * rho
        elif x == 1 and y == 1:
            val = 1.0 - rho
        else:
            val = 1.0
        return float(max(0.0, val))

    def calculate_match_matrix(self, lambda_: float, mu: float, rho: float = -0.05) -> np.ndarray:
        """
        Genera la matriz de probabilidades conjuntas de marcadores exactos P(X=x, Y=y).
        lambda_ = Goles esperados del equipo local (> 0)
        mu = Goles esperados del equipo visitante (> 0)
        rho = Parámetro de dependencia (-0.2 <= rho <= 0.2)
        """
        # Asegurar parámetros positivos
        lam = max(float(lambda_), 1e-4)
        m = max(float(mu), 1e-4)
        r = float(np.clip(rho, -0.25, 0.25))

        # Distribuciones Poisson independientes truncadas
        home_probs = np.array([poisson.pmf(i, lam) for i in range(self.max_goals)], dtype=float)
        away_probs = np.array([poisson.pmf(j, m) for j in range(self.max_goals)], dtype=float)

        # Matriz bivariada producto cartesiano
        matrix = np.outer(home_probs, away_probs)

        # Aplicar el ajuste de Dixon-Coles en las celdas bajas
        for x in range(min(2, self.max_goals)):
            for y in range(min(2, self.max_goals)):
                matrix[x, y] *= self.tau_adjustment(x, y, lam, m, r)

        # Limitar a valores válidos no negativos
        matrix = np.maximum(matrix, 0.0)

        # Normalizar para garantizar suma 1.0 exacta sobre el espacio muestral modelado
        total_sum = np.sum(matrix)
        if total_sum > 0:
            matrix = matrix / total_sum

        return matrix

    def generate_score_matrix(self, lambda_: float, mu: float, rho: float = -0.05) -> np.ndarray:
        """Alias para calculate_match_matrix requerido por simuladores tácticos."""
        return self.calculate_match_matrix(lambda_, mu, rho)

    def get_1x2_probabilities(self, match_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Agrega la matriz de marcadores en probabilidades del mercado 1X2.
        - Victoria Local (1): suma del triángulo inferior estricto (x > y)
        - Empate (X): suma de la diagonal principal (x == y)
        - Victoria Visitante (2): suma del triángulo superior estricto (y > x)
        """
        prob_home = float(np.tril(match_matrix, -1).sum())
        prob_draw = float(np.trace(match_matrix))
        prob_away = float(np.triu(match_matrix, 1).sum())
        return prob_home, prob_draw, prob_away

    def get_over_under_probabilities(self, match_matrix: np.ndarray, line: float = 2.5) -> Dict[str, float]:
        """Calcula probabilidades para el mercado Más/Menos goles (Over/Under)."""
        n, m = match_matrix.shape
        under_prob = 0.0
        for i in range(n):
            for j in range(m):
                if i + j < line:
                    under_prob += match_matrix[i, j]
        over_prob = max(0.0, 1.0 - under_prob)
        return {
            f"under_{line}": round(float(under_prob), 4),
            f"over_{line}": round(float(over_prob), 4)
        }

    def get_btts_probabilities(self, match_matrix: np.ndarray) -> Dict[str, float]:
        """Calcula probabilidad de Ambos Equipos Marcan (BTTS: Yes / No)."""
        # No marcan ambos si el local no marca (x=0) o el visitante no marca (y=0)
        p_no_home = np.sum(match_matrix[0, :])
        p_no_away = np.sum(match_matrix[:, 0])
        p_0_0 = match_matrix[0, 0]
        p_no = float(p_no_home + p_no_away - p_0_0)
        p_yes = max(0.0, 1.0 - p_no)
        return {
            "btts_yes": round(float(p_yes), 4),
            "btts_no": round(float(p_no), 4)
        }

    def get_most_likely_scores(self, match_matrix: np.ndarray, top_k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        """Devuelve los marcadores exactos más probables ordenados descendentemente."""
        scores = []
        n, m = match_matrix.shape
        for i in range(n):
            for j in range(m):
                scores.append(((i, j), float(match_matrix[i, j])))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def get_expected_goals(self, match_matrix: np.ndarray) -> Tuple[float, float]:
        """Calcula los goles esperados marginales derivados de la matriz."""
        n, m = match_matrix.shape
        home_xg = float(np.sum(match_matrix * np.arange(n)[:, None]))
        away_xg = float(np.sum(match_matrix * np.arange(m)[None, :]))
        return home_xg, away_xg
