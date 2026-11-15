from typing import Dict, Any, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from src.models_analytics.poisson_model import DixonColesModel


class PoissonOptimizer:
    """
    Motor matemático de Calibración por Máxima Verosimilitud (MLE) para modelos Poisson Bivariados.
    Resuelve el problema no lineal de optimización con restricciones de identificabilidad
    (sum(alpha_i) = 0) mediante el algoritmo Sequential Least Squares Programming (SLSQP).
    Totalmente desacoplado de bases de datos relacionales; opera sobre DataFrames en memoria.
    """

    def __init__(self):
        self.teams = np.array([])
        self.team_to_idx: Dict[str, int] = {}
        self.params: Dict[str, Any] = {}
        self.converged: bool = False
        self.iterations: int = 0
        self.log_likelihood: float = 0.0

    def _dixon_coles_log_likelihood(self, params: np.ndarray, df: pd.DataFrame, weights: Optional[np.ndarray] = None) -> float:
        """
        Función objetivo: Log-verosimilitud negativa del modelo Dixon-Coles con protección
        numérica contra overflow y underflow en regiones de gradiente extremo.
        """
        n_teams = len(self.teams)
        attack = params[:n_teams]
        defense = params[n_teams:2 * n_teams]
        home_adv = params[2 * n_teams]
        rho = params[2 * n_teams + 1]

        # Índices vectorizados
        home_idx = df['home_idx'].values
        away_idx = df['away_idx'].values
        hg = df['home_goals'].values
        ag = df['away_goals'].values

        # Protección numérica contra desbordamiento en función exponencial
        log_lambda = np.clip(attack[home_idx] + defense[away_idx] + home_adv, -10.0, 10.0)
        log_mu = np.clip(attack[away_idx] + defense[home_idx], -10.0, 10.0)

        lambda_ = np.exp(log_lambda)
        mu = np.exp(log_mu)

        # Probabilidades base independientes de Poisson
        prob_base = poisson.pmf(hg, lambda_) * poisson.pmf(ag, mu)

        # Ajuste Dixon-Coles vectorizado
        tau = np.ones(len(hg), dtype=float)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        tau[m00] = 1.0 - lambda_[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lambda_[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho

        # Forzar tau no-negativo
        tau = np.maximum(tau, 1e-6)

        # Probabilidad conjunta final acotada
        probs = prob_base * tau
        probs = np.where(np.isnan(probs) | (probs <= 0), 1e-12, probs)
        probs = np.clip(probs, 1e-12, 1.0)

        # Ponderación temporal si existe
        if weights is not None:
            nll = -np.sum(weights * np.log(probs))
        else:
            nll = -np.sum(np.log(probs))

        if not np.isfinite(nll):
            return 1e10
        return float(nll)

    def fit(self, df_matches: pd.DataFrame, weights: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """
        Calibra los parámetros de ataque (alpha), defensa (beta), ventaja local (gamma)
        y dependencia (rho) a partir del histórico de partidos.
        
        df_matches requiere columnas:
          - 'home_team': str
          - 'away_team': str
          - 'home_goals': int / float
          - 'away_goals': int / float
        """
        required_cols = {'home_team', 'away_team', 'home_goals', 'away_goals'}
        if not required_cols.issubset(df_matches.columns):
            missing = required_cols - set(df_matches.columns)
            raise ValueError(f"df_matches carece de columnas obligatorias: {missing}")

        if df_matches.empty:
            raise ValueError("df_matches está vacío, se requieren datos para la optimización.")

        self.teams = np.unique(df_matches[['home_team', 'away_team']].values)
        n_teams = len(self.teams)
        if n_teams < 2:
            raise ValueError("Se requieren al menos 2 equipos distintos para ajustar el modelo.")

        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}

        df = df_matches.copy()
        df['home_idx'] = df['home_team'].map(self.team_to_idx)
        df['away_idx'] = df['away_team'].map(self.team_to_idx)

        # Ponderaciones
        w_array = None
        if weights is not None:
            w_array = np.asarray(weights, dtype=float)
        elif 'weight' in df.columns:
            w_array = df['weight'].values.astype(float)

        # Vector inicial: [Ataques (n), Defensas (n), Ventaja Local (1), Rho (1)]
        initial_params = np.zeros(2 * n_teams + 2)
        initial_params[2 * n_teams] = 0.2  # Home advantage inicial típica en fútbol (~22%)
        initial_params[2 * n_teams + 1] = -0.05  # Rho inicial levemente negativo (empates 0-0/1-1)

        # Restricción de identificabilidad: sum(alpha_i) = 0
        constraints = [{'type': 'eq', 'fun': lambda x: sum(x[:n_teams])}]

        # Límites numéricamente estables:
        # Attack y Defense acotados en [-4.0, 4.0] para evitar divergencias
        # Ventaja local >= 0.0
        # Rho en [-0.2, 0.2]
        bounds = [(-4.0, 4.0)] * (2 * n_teams) + [(0.0, 3.0), (-0.2, 0.2)]

        # Minimización SLSQP
        result = minimize(
            self._dixon_coles_log_likelihood,
            initial_params,
            args=(df, w_array),
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-7}
        )

        raw_attack = result.x[:n_teams]
        raw_defense = result.x[n_teams:2 * n_teams]
        home_adv = float(max(0.0, result.x[2 * n_teams]))
        rho = float(np.clip(result.x[2 * n_teams + 1], -0.2, 0.2))

        # Re-centrado analítico exacto para asegurar identificabilidad estricta (sum=0.0)
        # La propiedad invariance mantiene lambda y mu idénticos
        mean_att = float(np.mean(raw_attack))
        attack = raw_attack - mean_att
        defense = raw_defense + mean_att

        self.converged = bool(result.success)
        self.iterations = int(result.nit)
        self.log_likelihood = float(-result.fun)

        self.params = {
            'attack': {t: float(attack[i]) for t, i in self.team_to_idx.items()},
            'defense': {t: float(defense[i]) for t, i in self.team_to_idx.items()},
            'home_adv': home_adv,
            'rho': rho
        }
        return self.params

    def predict_match(self, home_team: str, away_team: str, max_goals: int = 6) -> Dict[str, Any]:
        """
        Calcula las expectativas lambda, mu y probabilidades 1X2 para un enfrentamiento
        utilizando los parámetros calibrados.
        """
        if not self.params:
            raise ValueError("El modelo debe ser calibrado con fit() antes de realizar predicciones.")

        mean_defense = float(np.mean(list(self.params['defense'].values()))) if self.params['defense'] else 0.0

        att_h = self.params['attack'].get(home_team, 0.0)
        def_a = self.params['defense'].get(away_team, mean_defense)
        att_a = self.params['attack'].get(away_team, 0.0)
        def_h = self.params['defense'].get(home_team, mean_defense)
        home_adv = self.params['home_adv']
        rho = self.params['rho']

        lambda_ = float(np.exp(np.clip(att_h + def_a + home_adv, -5.0, 5.0)))
        mu = float(np.exp(np.clip(att_a + def_h, -5.0, 5.0)))

        dc = DixonColesModel(max_goals=max_goals)
        mat = dc.calculate_match_matrix(lambda_, mu, rho)
        h, d, a = dc.get_1x2_probabilities(mat)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "lambda_home": round(lambda_, 3),
            "mu_away": round(mu, 3),
            "prob_home": round(h, 4),
            "prob_draw": round(d, 4),
            "prob_away": round(a, 4),
            "matrix": mat
        }
