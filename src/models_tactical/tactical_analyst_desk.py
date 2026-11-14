from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.models_scouting.squad_diagnostics import SquadDiagnostics
from src.models_scouting.recommender import ScoutingRecommender
from src.models_analytics.elo_model import EloSystem
from src.models_analytics.poisson_model import DixonColesModel
from src.models_tactical.lineup_simulator import TacticalSimulator
from src.models_tactical.matchup_analyzer import MatchupAnalyzer
from src.models_tactical.tactical_playmaker import TacticalPlaymaker


class TacticalAnalystDesk:
    """
    Mesa de Trabajo del Analista Deportivo ("Virtual Analyst Desk").
    Coordina e hila de extremo a extremo los motores de:
      - Diagnóstico de Plantilla (Fase 3)
      - Scouting Contextual en las 5 Grandes Ligas y Ligas Secundarias (Fase 3)
      - Modelos Probabilísticos Elo y Dixon-Coles (Fase 4)
      - Simulador de Fichajes "What-If" y Monte Carlo (Fase 5)
      - Optimizador de Alineaciones y Pizarra Táctica (Fase 5)
    """

    def __init__(
        self,
        df_players_metrics: pd.DataFrame,
        elo_system: Optional[EloSystem] = None,
        poisson_model: Optional[DixonColesModel] = None,
        league_avg_xg: float = 1.35,
    ):
        self.df_metrics = df_players_metrics.copy()
        self.elo_system = elo_system or EloSystem()
        self.poisson_model = poisson_model or DixonColesModel()
        self.league_avg_xg = league_avg_xg

        # Inicializar submódulos
        self.diagnostics = SquadDiagnostics(self.df_metrics)
        self.recommender = ScoutingRecommender(self.df_metrics)
        self.simulator = TacticalSimulator(
            poisson_model=self.poisson_model,
            elo_system=self.elo_system,
            league_avg_xg=self.league_avg_xg,
        )
        self.matchup_analyzer = MatchupAnalyzer(league_avg_xg=self.league_avg_xg)
        self.playmaker = TacticalPlaymaker()

    # -------------------------------------------------------------------------
    # 1. Auditoría y Diagnóstico de Plantilla para un Club de LaLiga
    # -------------------------------------------------------------------------
    def audit_club_squad(self, team_name: str, metric: str = "xg_p90") -> Dict[str, Any]:
        """
        Detecta la posición más débil del equipo frente al promedio de la liga
        y genera alertas de envejecimiento / sobrecarga de minutos.
        """
        deficits = self.diagnostics.analyze_team_deficits(team_name=team_name, metric=metric)
        weakest_position_row = deficits.iloc[0] if not deficits.empty else None

        # Alertas de plantilla (jugadores veteranos con más del 60% de minutos)
        risk_alerts = self.diagnostics.detect_squad_risk_alerts(team_name=team_name)

        return {
            "team_name": team_name,
            "metric_evaluated": metric,
            "deficits_table": deficits,
            "weakest_position": weakest_position_row["position"] if weakest_position_row is not None else None,
            "position_deficit": round(float(weakest_position_row["deficit"]), 4) if weakest_position_row is not None else 0.0,
            "veteran_risk_alerts": risk_alerts,
        }

    # -------------------------------------------------------------------------
    # 2. Scouting Multiliga de Candidatos (Top 5 + Ligas de Talento)
    # -------------------------------------------------------------------------
    def scout_transfer_targets(
        self,
        target_position: str,
        features: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        leagues_filter: Optional[List[str]] = None,
        max_age: int = 27,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """
        Busca candidatos idóneos para reforzar la demarcación deficitaria
        explorando las 5 Grandes Ligas y ligas de talento secundarias.
        """
        df_pool = self.df_metrics[self.df_metrics["is_reliable"] == True].copy()

        # Filtrar posición
        if "position" in df_pool.columns:
            df_pool = df_pool[df_pool["position"] == target_position]

        # Filtrar ligas si aplica
        if leagues_filter and "league" in df_pool.columns:
            df_pool = df_pool[df_pool["league"].isin(leagues_filter)]

        # Filtrar edad si está presente
        if "age" in df_pool.columns:
            df_pool = df_pool[df_pool["age"] <= max_age]

        if df_pool.empty:
            return pd.DataFrame()

        # Calcular ranking ponderado
        weights = weights or {"xg_p90": 1.0, "xa_p90": 0.8, "interceptions_p90": 0.4}
        sub_recommender = ScoutingRecommender(df_pool)
        ranked_candidates = sub_recommender.calculate_weighted_score(weights)

        cols_to_show = [c for c in [
            "player_id", "player_name", "team_name", "league", "position",
            "age", "xg_p90", "xa_p90", "interceptions_p90", "custom_score"
        ] if c in ranked_candidates.columns]

        return ranked_candidates[cols_to_show].head(top_k)

    # -------------------------------------------------------------------------
    # 3. Simulación de Fichaje ("What-If") y Proyección de Puntos
    # -------------------------------------------------------------------------
    def simulate_transfer_scenario(
        self,
        team_name: str,
        player_out_id: int,
        candidate_player_id: int,
        base_lambda: float = 1.30,
        base_mu: float = 1.15,
        remaining_matches: int = 15,
        current_points: int = 42,
    ) -> Dict[str, Any]:
        """
        Evalúa el impacto integral de sustituir a un jugador titular por un fichaje recomendado:
        actualiza lambda, mu, xPts y proyecta la temporada vía Monte Carlo.
        """
        current_squad = self.df_metrics[self.df_metrics.get("team_name", "") == team_name]
        candidate_series = self.df_metrics[self.df_metrics["player_id"] == candidate_player_id]

        if candidate_series.empty:
            raise ValueError(f"Candidato ID {candidate_player_id} no encontrado en la base de datos.")

        cand_row = candidate_series.iloc[0]
        cand_stats = {
            "xg_p90": float(cand_row.get("xg_p90", 0.0)),
            "xa_p90": float(cand_row.get("xa_p90", 0.0)),
            "interceptions_p90": float(cand_row.get("interceptions_p90", 0.0)),
        }

        # Ejecutar evaluación analítica del intercambio
        swap_eval = self.simulator.evaluate_player_swap(
            current_squad_df=current_squad,
            player_out_id=player_out_id,
            candidate_stats=cand_stats,
            base_lambda=base_lambda,
            base_mu=base_mu,
            candidate_origin_team=cand_row.get("team_name"),
            destination_team=team_name,
        )

        # Simular proyección estocástica Monte Carlo de las jornadas restantes
        mc_eval = self.simulator.simulate_monte_carlo_season(
            base_lambda=base_lambda,
            base_mu=base_mu,
            new_lambda=swap_eval["new_lambda"],
            new_mu=swap_eval["new_mu"],
            remaining_matches=remaining_matches,
            current_points=current_points,
        )

        return {
            "player_out_name": swap_eval["player_out_name"],
            "candidate_name": cand_row.get("player_name"),
            "candidate_origin_team": cand_row.get("team_name"),
            "candidate_league": cand_row.get("league", "Desconocida"),
            "swap_impact": swap_eval,
            "monte_carlo_projection": mc_eval,
        }

    # -------------------------------------------------------------------------
    # 4. Preparación Táctica de Partido (Alineación Óptima y Pizarra vs Rival)
    # -------------------------------------------------------------------------
    def prepare_match_plan(
        self,
        team_name: str,
        opponent_profile: Dict[str, float],
        formation: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Prepara el partido de la próxima jornada seleccionando el XI más idóneo
        para contrarrestar al rival y redacta las directrices tácticas de pizarra.
        """
        formation = formation or {"GK": 1, "DF": 4, "MF": 3, "FW": 3}
        team_players = self.df_metrics[self.df_metrics.get("team_name", "") == team_name]

        # Recomendar XI titular óptimo
        optimal_lineup = self.matchup_analyzer.recommend_optimal_lineup(
            available_players_df=team_players,
            formation=formation,
            opponent_profile=opponent_profile,
        )

        # Radiografía del enfrentamiento
        matchup_insights = self.matchup_analyzer.analyze_tactical_matchup(
            lineup_df=optimal_lineup,
            opponent_profile=opponent_profile,
        )

        # Plan de jugadas recomendadas
        plays = self.playmaker.suggest_tactical_plays(
            lineup_df=optimal_lineup,
            opponent_profile=opponent_profile,
        )

        return {
            "team_name": team_name,
            "optimal_lineup": optimal_lineup,
            "matchup_insights": matchup_insights,
            "tactical_plays": plays,
        }
