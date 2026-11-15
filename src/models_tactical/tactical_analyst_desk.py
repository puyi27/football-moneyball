from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.models_scouting.squad_diagnostics import SquadDiagnostics
from src.models_scouting.recommender import ScoutingRecommender
from src.models_analytics.elo_model import EloSystem
from src.models_analytics.poisson_model import DixonColesModel
from src.models_analytics.team_profiler import TeamProfiler
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
      - Optimizador de Alineaciones, Pizarra Táctica y Duelos Individuales (Fase 5 y 7)
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
        self.team_profiler = TeamProfiler(df_players=self.df_metrics, elo_system=self.elo_system)
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
        opponent_profile: Any = None,
        formation: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Prepara el partido de la próxima jornada seleccionando el XI más idóneo
        para contrarrestar al rival y redacta las directrices tácticas de pizarra.
        Consume el perfil táctico oficial de TeamProfiler y calcula duelos individuales.
        Acepta en opponent_profile tanto un diccionario de atributos como el nombre del club rival (str).
        """
        opp_name = "Rival"
        if isinstance(opponent_profile, str):
            opp_name = opponent_profile
            # Sintetizar perfil táctico a partir del rival y su Elo
            rival_elo = self.elo_system.get_rating(opp_name) if hasattr(self.elo_system, "get_rating") else 1650.0
            my_elo = self.elo_system.get_rating(team_name) if hasattr(self.elo_system, "get_rating") else 1650.0
            elo_diff = rival_elo - my_elo

            # Consumir el perfil táctico oficial de TeamProfiler sin duplicar cálculos
            team_adn = None
            alpha_power = 1.05
            beta_leak = 1.05
            try:
                overview = self.team_profiler.get_team_overview(opp_name)
                team_adn = overview.get("tactical_adn", {})
                alpha_power = float(overview.get("alpha_offensive_power", 1.1))
                beta_leak = float(overview.get("beta_defensive_leak", 1.0))
            except Exception:
                team_adn = TeamProfiler.TEAM_TACTICAL_STYLES.get(opp_name, {})
                alpha_power = 1.25 if elo_diff > 100 else (0.85 if elo_diff < -100 else 1.05)
                beta_leak = 0.85 if elo_diff > 100 else (1.25 if elo_diff < -100 else 1.05)

            press_val = float(team_adn.get("pressing_intensity", 7.5)) if team_adn else 7.5
            poss_val = float(team_adn.get("possession_pct", 52.0)) if team_adn else 52.0
            style_name = str(team_adn.get("style", "Bloque Competitivo")) if team_adn else "Bloque Competitivo"
            pref_channel = str(team_adn.get("preferred_channel", "Juego Exterior")) if team_adn else "Juego Exterior"

            if elo_diff > 100:
                profile_dict = {
                    "defense_weakness": round(float(beta_leak * 0.9), 2),
                    "attack_threat": round(float(alpha_power * 1.1), 2),
                    "high_line_vulnerability": round(float(np.clip(press_val / 10.0, 0.4, 0.9)), 2),
                    "low_block_tendency": 0.20,
                    "pressing_intensity": press_val,
                }
                form_text = "5-3-2 Transición Rápida & Repliegue Asimétrico"
                mister_inst = f"El rival ({opp_name}) asfixia con {style_name.lower()} ({rival_elo:.0f} Elo, {poss_val:.1f}% posesión). Priorizar contención en {pref_channel}, ceder la iniciativa territorial, cerrar pasillos interiores y castigar con diagonales a la espalda de sus centrales."
            elif elo_diff < -100:
                profile_dict = {
                    "defense_weakness": round(float(beta_leak * 1.15), 2),
                    "attack_threat": round(float(alpha_power * 0.85), 2),
                    "high_line_vulnerability": 0.30,
                    "low_block_tendency": 0.80,
                    "pressing_intensity": press_val,
                }
                form_text = "4-3-3 Ofensivo con Amplitud Total y Sobrecarga"
                mister_inst = f"{opp_name} plantará un bloque bajo ultradefensivo. Fijar con laterales en amplitud máxima, paciencia en circulación y buscar centros tensos al punto de penalti y disparos desde la frontal."
            else:
                profile_dict = {
                    "defense_weakness": round(float(beta_leak), 2),
                    "attack_threat": round(float(alpha_power), 2),
                    "high_line_vulnerability": round(float(np.clip(press_val / 10.0, 0.35, 0.75)), 2),
                    "low_block_tendency": 0.45,
                    "pressing_intensity": press_val,
                }
                form_text = "4-2-3-1 Dinámico con Presión Tras Pérdida"
                mister_inst = f"Duelo de fuerzas parejas frente a {opp_name} ({style_name}). Clave ganar los duelos en la medular, gobernar segundas jugadas y buscar desborde en el 1 contra 1 por las bandas."
        elif isinstance(opponent_profile, dict):
            profile_dict = opponent_profile
            form_text = "4-3-3 Adaptado al Análisis de Pizarra"
            mister_inst = "Explotar las debilidades defensivas detectadas y neutralizar las vías preferentes de progresión del adversario."
        else:
            profile_dict = {"defense_weakness": 1.0, "attack_threat": 1.0, "high_line_vulnerability": 0.5, "low_block_tendency": 0.5}
            form_text = "4-3-3 Estándar de Competición"
            mister_inst = "Mantener rigor posicional, transiciones ágiles y equilibrio entre líneas."

        formation = formation or {"GK": 1, "DF": 4, "MF": 3, "FW": 3}
        team_players = self.df_metrics[self.df_metrics.get("team_name", "") == team_name]
        if team_players.empty:
            team_players = self.df_metrics.head(18)

        # Recomendar XI titular óptimo
        optimal_lineup = self.matchup_analyzer.recommend_optimal_lineup(
            available_players_df=team_players,
            formation=formation,
            opponent_profile=profile_dict,
        )

        # Radiografía del enfrentamiento colectivo
        matchup_insights = self.matchup_analyzer.analyze_tactical_matchup(
            lineup_df=optimal_lineup,
            opponent_profile=profile_dict,
        )

        # Buscar XI titular rival disponible para duelos directos hombre a hombre
        rival_pool = self.df_metrics[self.df_metrics.get("team_name", "") == opp_name]
        rival_xi = None
        if not rival_pool.empty and len(rival_pool) >= 11:
            rival_xi = self.matchup_analyzer.recommend_optimal_lineup(
                available_players_df=rival_pool,
                formation=formation,
                opponent_profile={"defense_weakness": 1.0, "attack_threat": 1.0},
            )

        # Simular los 5 duelos individuales decisivos
        individual_duels = self.matchup_analyzer.simulate_individual_duels(
            local_lineup=optimal_lineup,
            rival_lineup=rival_xi,
            opponent_profile=profile_dict,
        )

        # Evaluar la química y sinergia táctica del Once
        synergy_eval = self.simulator.evaluate_lineup_synergy(
            lineup_df=optimal_lineup,
            formation=form_text,
        )

        # Plan de jugadas recomendadas
        plays = self.playmaker.suggest_tactical_plays(
            lineup_df=optimal_lineup,
            opponent_profile=profile_dict,
        )

        return {
            "team_name": team_name,
            "opponent_name": opp_name,
            "formation_recommended": form_text,
            "tactical_instructions": mister_inst,
            "optimal_lineup": optimal_lineup,
            "matchup_insights": matchup_insights,
            "individual_duels": individual_duels,
            "synergy_assessment": synergy_eval,
            "tactical_plays": plays,
        }

    # -------------------------------------------------------------------------
    # 5. Simulación Integral de Jornada de Partido ("Match Day Experience")
    # -------------------------------------------------------------------------
    def simulate_match_day(
        self,
        home_team: str,
        away_team: str,
        home_formation: Optional[Dict[str, int]] = None,
        away_formation: Optional[Dict[str, int]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Orquesta una jornada completa de partido ("Match Day Experience"):
          1. Prepara planes tácticos y Onces titulares para ambos clubes.
          2. Analiza los micro-duelos tácticos hombre a hombre.
          3. Predice el resultado 1X2 con modelos calibrados.
          4. Ejecuta el gemelo digital (Digital Twin) estocástico minuto a minuto (1-90).
        """
        from src.models_tactical.match_scenario_simulator import MatchScenarioSimulator
        scenario_sim = MatchScenarioSimulator(poisson_model=self.poisson_model)

        plan_home = self.prepare_match_plan(home_team, away_team, formation=home_formation)
        plan_away = self.prepare_match_plan(away_team, home_team, formation=away_formation)

        home_xi = plan_home["optimal_lineup"]
        away_xi = plan_away["optimal_lineup"]

        h_elo = self.elo_system.get_rating(home_team) if hasattr(self.elo_system, "get_rating") else 1650.0
        a_elo = self.elo_system.get_rating(away_team) if hasattr(self.elo_system, "get_rating") else 1650.0

        pred_1x2 = self.matchup_analyzer.predict_match_outcome(
            home_team=home_team,
            away_team=away_team,
            local_lineup=home_xi,
            rival_lineup=away_xi,
            home_elo=h_elo,
            away_elo=a_elo,
        )

        h2h_duels = self.matchup_analyzer.simulate_individual_duels(
            local_lineup=home_xi,
            rival_lineup=away_xi,
        )

        timeline_out = scenario_sim.simulate_match_timeline(
            home_team=home_team,
            away_team=away_team,
            base_elo_home=h_elo,
            base_elo_away=a_elo,
            seed=seed,
            individual_duels=h2h_duels,
        )

        return {
            "matchup": f"{home_team} vs {away_team}",
            "home_plan": plan_home,
            "away_plan": plan_away,
            "head_to_head_duels": h2h_duels,
            "outcome_prediction_1x2": pred_1x2,
            "match_timeline": timeline_out,
        }
