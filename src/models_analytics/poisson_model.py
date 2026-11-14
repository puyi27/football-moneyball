import numpy as np
from scipy.stats import poisson

class DixonColesModel:
    def __init__(self, max_goals: int = 5):
        self.max_goals = max_goals

    def tau_adjustment(self, x: int, y: int, lambda_: float, mu: float, rho: float) -> float:
        """
        Calcula el factor de corrección de Dixon y Coles para bajas anotaciones.
        Incrementa la probabilidad de 0-0, 1-1, 1-0 y 0-1.
        """
        if x == 0 and y == 0:
            return 1 - lambda_ * mu * rho
        elif x == 0 and y == 1:
            return 1 + lambda_ * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0

    def calculate_match_matrix(self, lambda_: float, mu: float, rho: float) -> np.ndarray:
        """
        Genera la matriz de probabilidades de marcadores exactos.
        lambda_ = Goles esperados Local
        mu = Goles esperados Visitante
        rho = Parámetro de dependencia
        """
        # Calcular distribuciones Poisson independientes truncadas
        home_probs = np.array([poisson.pmf(i, lambda_) for i in range(self.max_goals)])
        away_probs = np.array([poisson.pmf(i, mu) for i in range(self.max_goals)])
        
        # Crear matriz bivariada base (asumiendo independencia)
        matrix = np.outer(home_probs, away_probs)

        # Aplicar el ajuste de Dixon-Coles
        for x in range(2):
            for y in range(2):
                matrix[x, y] *= self.tau_adjustment(x, y, lambda_, mu, rho)

        # Normalizar para garantizar suma 1.0 (corrigiendo el truncamiento en max_goals)
        matrix = matrix / np.sum(matrix)
        
        return matrix

    def get_1x2_probabilities(self, match_matrix: np.ndarray) -> tuple:
        """Agrega la matriz en probabilidades de mercado 1X2."""
        prob_home = np.tril(match_matrix, -1).sum()
        prob_draw = np.trace(match_matrix)
        prob_away = np.triu(match_matrix, 1).sum()
        return float(prob_home), float(prob_draw), float(prob_away)