from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from src.models_analytics.poisson_model import DixonColesModel
from src.models_analytics.elo_model import EloSystem


class TacticalSimulator:
    """
    Simulador de Impacto ("What-If" Engine) para directores deportivos y analistas tácticos.
    Conecta el rendimiento individual de jugadores (scouting) con los parámetros colectivos de
    gol esperado (lambda, mu) del modelo Dixon-Coles y valoraciones de fuerza Elo.
    """

    def __init__(
        self,
        poisson_model: Optional[Any] = None,
        elo_system: Optional[EloSystem] = None,
        league_avg_xg: float = 1.35,
    ):
        self.poisson_model = poisson_model or DixonColesModel()
        self.elo_system = elo_system
        self.league_avg_xg = max(league_avg_xg, 0.01)

    def evaluate_player_swap(
        self,
        current_squad_df: pd.DataFrame,
        player_out_id: int,
        candidate_stats: Dict[str, float],
        base_lambda: float,
        base_mu: float,
        rho: float = -0.05,
        candidate_origin_team: Optional[str] = None,
        destination_team: Optional[str] = None,
        offensive_weight: float = 0.35,
        defensive_weight: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Evalúa el impacto de reemplazar un jugador titular por un fichaje o sustituto.
        Calcula la variación en lambda (ataque), mu (defensa rival), probabilidad 1X2 y puntos esperados (xPts).
        """
        if player_out_id not in current_squad_df["player_id"].values:
            raise ValueError(f"El jugador con ID {player_out_id} no existe en la plantilla actual.")

        player_out = current_squad_df[current_squad_df["player_id"] == player_out_id].iloc[0]

        # 1. Factor de ajuste por contexto competitivo (Elo origin / Elo destination)
        context_factor = 1.0
        if self.elo_system and candidate_origin_team and destination_team:
            elo_origin = self.elo_system.get_rating(candidate_origin_team)
            elo_dest = self.elo_system.get_rating(destination_team)
            if elo_dest > 0:
                # Limitamos el factor de contexto entre 0.70 y 1.30 para regularidad estadística
                context_factor = float(np.clip(elo_origin / elo_dest, 0.70, 1.30))

        # 2. Métricas ofensivas individuales (xG + xA) con ajuste contextual
        out_off = float(player_out.get("xg_p90", 0.0) + player_out.get("xa_p90", 0.0))
        in_off = float(candidate_stats.get("xg_p90", 0.0) + candidate_stats.get("xa_p90", 0.0)) * context_factor
        delta_off = (in_off - out_off) / self.league_avg_xg

        # 3. Métricas defensivas individuales (intercepciones y recuperaciones)
        out_def = float(player_out.get("interceptions_p90", 0.0))
        in_def = float(candidate_stats.get("interceptions_p90", 0.0)) * context_factor
        ref_def = max(float(current_squad_df.get("interceptions_p90", pd.Series([1.0])).mean()), 0.1)
        delta_def = (in_def - out_def) / ref_def

        # 4. Actualización proporcional de tasas de gol esperado (lambda y mu)
        new_lambda = max(0.1, base_lambda * (1.0 + (offensive_weight * delta_off)))
        new_mu = max(0.1, base_mu * (1.0 - (defensive_weight * delta_def)))

        # 5. Cálculo de probabilidades 1X2 con el modelo Dixon-Coles
        base_probs = self._calculate_1x2_probs(base_lambda, base_mu, rho)
        new_probs = self._calculate_1x2_probs(new_lambda, new_mu, rho)

        # 6. Puntos esperados por partido (xPts = 3 * P(Win) + 1 * P(Draw))
        base_xpts = (base_probs["win"] * 3.0) + (base_probs["draw"] * 1.0)
        new_xpts = (new_probs["win"] * 3.0) + (new_probs["draw"] * 1.0)
        delta_xpts = new_xpts - base_xpts

        return {
            "player_out_name": player_out.get("player_name", f"ID {player_out_id}"),
            "context_factor": round(context_factor, 3),
            "base_lambda": round(float(base_lambda), 4),
            "new_lambda": round(float(new_lambda), 4),
            "delta_lambda": round(float(new_lambda - base_lambda), 4),
            "base_mu": round(float(base_mu), 4),
            "new_mu": round(float(new_mu), 4),
            "delta_mu": round(float(new_mu - base_mu), 4),
            "base_win_prob": round(float(base_probs["win"]), 4),
            "new_win_prob": round(float(new_probs["win"]), 4),
            "win_prob_delta": round(float(new_probs["win"] - base_probs["win"]), 4),
            "base_xpts_per_match": round(float(base_xpts), 4),
            "new_xpts_per_match": round(float(new_xpts), 4),
            "xpts_gain_per_match": round(float(delta_xpts), 4),
            "xpts_gain_38_matches": round(float(delta_xpts * 38.0), 2),
        }

    def simulate_monte_carlo_season(
        self,
        base_lambda: float,
        base_mu: float,
        new_lambda: float,
        new_mu: float,
        remaining_matches: int = 15,
        current_points: int = 40,
        rho: float = -0.05,
        n_simulations: int = 5000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Ejecuta N simulaciones Monte Carlo de las jornadas restantes para proyectar
        distribuciones de puntos finales, probabilidades de clasificar a Europa (>=60 pts)
        o descenso (<=38 pts).
        """
        np.random.seed(seed)

        base_mat = self._get_match_matrix(base_lambda, base_mu, rho)
        new_mat = self._get_match_matrix(new_lambda, new_mu, rho)

        # Probabilidades planas de marcadores
        flat_base = base_mat.flatten()
        flat_new = new_mat.flatten()
        flat_base /= flat_base.sum()
        flat_new /= flat_new.sum()

        max_goals = base_mat.shape[0]
        # Puntos otorgados por cada celda de la matriz (i = goles local, j = goles visitante)
        scores_points = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                if i > j:
                    scores_points[i, j] = 3
                elif i == j:
                    scores_points[i, j] = 1
                else:
                    scores_points[i, j] = 0
        points_flat = scores_points.flatten()

        # Simular jornadas restantes
        sim_base_indices = np.random.choice(len(flat_base), size=(n_simulations, remaining_matches), p=flat_base)
        sim_new_indices = np.random.choice(len(flat_new), size=(n_simulations, remaining_matches), p=flat_new)

        base_points_gained = points_flat[sim_base_indices].sum(axis=1)
        new_points_gained = points_flat[sim_new_indices].sum(axis=1)

        base_total = current_points + base_points_gained
        new_total = current_points + new_points_gained

        return {
            "n_simulations": n_simulations,
            "remaining_matches": remaining_matches,
            "current_points": current_points,
            "base_projected_points_mean": round(float(np.mean(base_total)), 2),
            "new_projected_points_mean": round(float(np.mean(new_total)), 2),
            "points_gain_mean": round(float(np.mean(new_total) - np.mean(base_total)), 2),
            "prob_gain_points": round(float(np.mean(new_total > base_total)), 4),
            "base_prob_europe_60pts": round(float(np.mean(base_total >= 60)), 4),
            "new_prob_europe_60pts": round(float(np.mean(new_total >= 60)), 4),
            "europe_prob_delta": round(float(np.mean(new_total >= 60) - np.mean(base_total >= 60)), 4),
            "base_prob_relegation_38pts": round(float(np.mean(base_total <= 38)), 4),
            "new_prob_relegation_38pts": round(float(np.mean(new_total <= 38)), 4),
        }

    def _get_match_matrix(self, lam: float, mu: float, rho: float) -> np.ndarray:
        """Invoca la matriz de resultados según la interfaz del modelo inyectado."""
        if hasattr(self.poisson_model, "calculate_match_matrix"):
            return self.poisson_model.calculate_match_matrix(lam, mu, rho)
        elif hasattr(self.poisson_model, "generate_score_matrix"):
            return self.poisson_model.generate_score_matrix(lam, mu)
        else:
            from scipy.stats import poisson
            h = np.array([poisson.pmf(i, lam) for i in range(6)])
            a = np.array([poisson.pmf(j, mu) for j in range(6)])
            mat = np.outer(h, a)
            return mat / mat.sum()

    def _calculate_1x2_probs(self, lam: float, mu: float, rho: float = -0.05) -> Dict[str, float]:
        """Calcula probabilidades 1X2 compatibles con DixonColesModel o matrices genéricas."""
        matrix = self._get_match_matrix(lam, mu, rho)
        if hasattr(self.poisson_model, "get_1x2_probabilities"):
            h, d, a = self.poisson_model.get_1x2_probabilities(matrix)
            return {"win": float(h), "draw": float(d), "loss": float(a)}
        else:
            win = float(np.sum(np.tril(matrix, -1)))
            draw = float(np.sum(np.diag(matrix)))
            loss = float(np.sum(np.triu(matrix, 1)))
            return {"win": win, "draw": draw, "loss": loss}
