import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

class PoissonOptimizer:
    def __init__(self):
        self.teams = []
        self.team_to_idx = {}
        self.params = {}

    def _dixon_coles_log_likelihood(self, params, df):
        """Función de coste: Log-verosimilitud negativa del modelo Dixon-Coles."""
        n_teams = len(self.teams)
        attack = params[:n_teams]
        defense = params[n_teams:2*n_teams]
        home_adv = params[2*n_teams]
        rho = params[2*n_teams + 1]

        # Índices vectorizados para acceso rápido
        home_idx = df['home_idx'].values
        away_idx = df['away_idx'].values
        hg = df['home_goals'].values
        ag = df['away_goals'].values

        # Expectativas Poisson (lambda para local, mu para visitante)
        # Se usa exp() para garantizar que los valores sean siempre positivos
        lambda_ = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
        mu = np.exp(attack[away_idx] + defense[home_idx])

        # Probabilidades base independientes
        prob_base = poisson.pmf(hg, lambda_) * poisson.pmf(ag, mu)
        
        # Ajuste Dixon-Coles vectorizado
        tau = np.ones(len(hg))
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        tau[m00] = 1 - lambda_[m00] * mu[m00] * rho
        tau[m01] = 1 + lambda_[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho

        # Calcular probabilidad final (evitar log(0) con np.clip)
        probs = prob_base * tau
        probs = np.clip(probs, 1e-10, 1.0)
        
        # Devolver log-verosimilitud negativa (porque scipy minimiza, no maximiza)
        return -np.sum(np.log(probs))

    def fit(self, df_matches: pd.DataFrame) -> dict:
        """
        Calibra los parámetros de ataque y defensa según el histórico.
        df_matches requiere: home_team, away_team, home_goals, away_goals
        """
        self.teams = np.unique(df_matches[['home_team', 'away_team']].values)
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        n_teams = len(self.teams)

        df = df_matches.copy()
        df['home_idx'] = df['home_team'].map(self.team_to_idx)
        df['away_idx'] = df['away_team'].map(self.team_to_idx)

        # Vector inicial de parámetros: [Ataques (n), Defensas (n), Ventaja Local (1), Rho (1)]
        # Empezamos con parámetros neutrales (0 para exp() = 1) y pequeña ventaja local
        initial_params = np.zeros(2 * n_teams + 2)
        initial_params[2 * n_teams] = 0.2 

        # Restricción obligatoria: La suma de los ataques debe ser 0 para evitar infinitas soluciones equivalentes
        constraints = [{'type': 'eq', 'fun': lambda x: sum(x[:n_teams])}]
        
        # Límites (Bounds): Ventaja local > 0, Rho contenido entre -0.2 y 0.2 (típico en fútbol)
        bounds = [(None, None)] * (2 * n_teams) + [(0, None), (-0.2, 0.2)]

        # Minimización mediante Sequential Least Squares Programming
        result = minimize(
            self._dixon_coles_log_likelihood, 
            initial_params, 
            args=(df,), 
            method='SLSQP', 
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 150}
        )

        attack = result.x[:n_teams]
        defense = result.x[n_teams:2*n_teams]
        home_adv = result.x[2*n_teams]
        rho = result.x[2*n_teams + 1]

        self.params = {
            'attack': {t: float(attack[i]) for t, i in self.team_to_idx.items()},
            'defense': {t: float(defense[i]) for t, i in self.team_to_idx.items()},
            'home_adv': float(home_adv),
            'rho': float(rho)
        }
        return self.params