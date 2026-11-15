from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class MatchupAnalyzer:
    """
    Optimizador de Alineaciones y Matchups Tácticos frente a rivales específicos.
    Cruza las métricas p90 de la plantilla disponible con las debilidades y fortalezas
    del próximo adversario para seleccionar el XI inicial con mayor probabilidad de éxito.
    """

    DEFAULT_FORMATION = {"GK": 1, "DF": 4, "MF": 3, "FW": 3}

    def __init__(self, league_avg_xg: float = 1.35):
        self.league_avg_xg = league_avg_xg

    def calculate_player_tactical_fit(
        self,
        player_row: pd.Series,
        opponent_profile: Dict[str, float]
    ) -> float:
        """
        Calcula el Matchup Score de un jugador individual contra el perfil del rival.
        Ajusta la ponderación ofensiva vs defensiva según si el rival es vulnerable o dominante.
        """
        opp_def_weakness = opponent_profile.get("defense_weakness", 1.0)
        opp_att_threat = opponent_profile.get("attack_threat", 1.0)
        high_line_vuln = opponent_profile.get("high_line_vulnerability", 0.5)
        low_block = opponent_profile.get("low_block_tendency", 0.5)

        xg = float(player_row.get("xg_p90", 0.0) or 0.0)
        xa = float(player_row.get("xa_p90", 0.0) or 0.0)
        interceptions = float(player_row.get("interceptions_p90", 0.0) or 0.0)
        tackles = float(player_row.get("tackles_p90", 0.0) or 0.0)

        # Contribución ofensiva: Potenciada si el rival tiene debilidad defensiva
        # o si hay espacios (high line con xG de ruptura) o si hay que romper bloques con xA
        off_score = (xg * (1.0 + 0.5 * high_line_vuln) + xa * (1.0 + 0.5 * low_block)) * opp_def_weakness

        # Contribución defensiva: Potenciada si el rival genera alto volumen ofensivo (contención y repliegue)
        def_score = (interceptions * 0.7 + tackles * 0.3) * opp_att_threat

        # Ponderación según la posición del jugador
        pos = str(player_row.get("position", "MF")).upper()
        if "FW" in pos:
            fit_score = (off_score * 0.85) + (def_score * 0.15)
        elif "DF" in pos or "GK" in pos:
            fit_score = (off_score * 0.20) + (def_score * 0.80)
        else:  # Centrocampistas (MF)
            fit_score = (off_score * 0.55) + (def_score * 0.45)

        return float(round(fit_score, 4))

    def recommend_optimal_lineup(
        self,
        available_players_df: pd.DataFrame,
        formation: Optional[Dict[str, int]] = None,
        opponent_profile: Optional[Dict[str, float]] = None,
        require_reliable: bool = True
    ) -> pd.DataFrame:
        """
        Selecciona el mejor XI titular dadas las restricciones de formación y el rival.
        Garantiza una alineación de exactamente 11 futbolistas reales sin duplicidades.
        """
        formation = formation or self.DEFAULT_FORMATION
        opponent_profile = opponent_profile or {"defense_weakness": 1.0, "attack_threat": 1.0}

        df = available_players_df.copy()
        if require_reliable and "is_reliable" in df.columns:
            rel_df = df[df["is_reliable"] == True].copy()
            if not rel_df.empty and len(rel_df) >= sum(formation.values()):
                df = rel_df

        if df.empty:
            raise ValueError("No hay jugadores fiables disponibles para confeccionar la alineación.")

        # Calcular tactical fit score para cada jugador disponible
        df["tactical_fit_score"] = df.apply(
            lambda row: self.calculate_player_tactical_fit(row, opponent_profile), axis=1
        )

        selected_player_ids = set()
        selected_rows = []

        # 1. Asignar por posición preferente sin duplicar futbolistas
        for position, count in formation.items():
            pos_candidates = df[
                (df["position"] == position) & (~df["player_id"].isin(selected_player_ids))
            ].sort_values(by="tactical_fit_score", ascending=False)

            picked = pos_candidates.head(count)
            for _, r in picked.iterrows():
                selected_player_ids.add(r["player_id"])
                selected_rows.append(r)

            shortfall = count - len(picked)
            if shortfall > 0:
                # Buscar alternativas no seleccionadas (evitando GK para posiciones de campo)
                if position != "GK":
                    alt_candidates = df[
                        (df["position"] != "GK") & (~df["player_id"].isin(selected_player_ids))
                    ].sort_values(by="tactical_fit_score", ascending=False)
                else:
                    alt_candidates = df[
                        ~df["player_id"].isin(selected_player_ids)
                    ].sort_values(by="tactical_fit_score", ascending=False)

                alt_picked = alt_candidates.head(shortfall)
                for _, r in alt_picked.iterrows():
                    selected_player_ids.add(r["player_id"])
                    selected_rows.append(r)

        # 2. Garantizar que la alineación total alcance exactamente el número requerido (11)
        total_required = sum(formation.values())
        if len(selected_rows) < total_required and len(df) > len(selected_rows):
            missing_count = total_required - len(selected_rows)
            remaining_pool = df[~df["player_id"].isin(selected_player_ids)].sort_values(
                by="tactical_fit_score", ascending=False
            )
            extra_picked = remaining_pool.head(missing_count)
            for _, r in extra_picked.iterrows():
                selected_player_ids.add(r["player_id"])
                selected_rows.append(r)

        result_df = pd.DataFrame(selected_rows).drop_duplicates(subset=["player_id"])
        return result_df.sort_values(by="tactical_fit_score", ascending=False)

    def analyze_tactical_matchup(
        self,
        lineup_df: pd.DataFrame,
        opponent_profile: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Genera una radiografía del XI frente al rival: ventajas competitivas,
        zonas de riesgo y balance de fuerza.
        """
        avg_off = float(lineup_df.get("xg_p90", pd.Series([0.0])).mean() + lineup_df.get("xa_p90", pd.Series([0.0])).mean())
        avg_def = float(lineup_df.get("interceptions_p90", pd.Series([0.0])).mean())

        opp_def_vuln = opponent_profile.get("defense_weakness", 1.0)
        opp_att = opponent_profile.get("attack_threat", 1.0)

        insights = []
        if avg_off > 0.45 and opp_def_vuln >= 1.1:
            insights.append("Ventaja Ofensiva Clara: Capacidad de generar desequilibrio constante en último tercio.")
        elif avg_off < 0.25 and opp_def_vuln < 0.9:
            insights.append("Riesgo de Atasco: Rival con bloque férreo; se requerirá juego exterior o balón parado.")

        if avg_def >= 1.4 and opp_att >= 1.1:
            insights.append("Bloque Sólido: Nivel de intercepciones óptimo para contener el ataque rival.")
        elif avg_def < 1.0 and opp_att >= 1.2:
            insights.append("Alerta Defensiva: El rival tiene gran potencia ofensiva y el XI concede recuperaciones.")

        return {
            "xi_avg_offensive_threat": round(avg_off, 3),
            "xi_avg_defensive_recovery": round(avg_def, 3),
            "opponent_defense_weakness": round(opp_def_vuln, 3),
            "opponent_attack_threat": round(opp_att, 3),
            "tactical_insights": insights,
        }

    def predict_match_outcome(
        self,
        home_team: str,
        away_team: str,
        local_lineup: Optional[pd.DataFrame] = None,
        rival_lineup: Optional[pd.DataFrame] = None,
        home_advantage: float = 0.20,
        home_elo: Optional[float] = None,
        away_elo: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predice las probabilidades de resultado 1X2 enfrentando las alineaciones de ambos clubes.
        Garantiza que win_home_prob + draw_prob + win_away_prob sumen exactamente 1.0 (100%).
        Utilizado en el laboratorio 'What-If' del cuadro de mando interactivo (app.py).
        """
        if local_lineup is not None and not local_lineup.empty:
            home_off = float(local_lineup.get("xg_p90", pd.Series([0.35])).mean()) * 3.2
            home_def = float(local_lineup.get("interceptions_p90", pd.Series([1.2])).mean())
        else:
            home_off, home_def = 1.35, 1.2

        if rival_lineup is not None and not rival_lineup.empty:
            away_off = float(rival_lineup.get("xg_p90", pd.Series([0.35])).mean()) * 3.0
            away_def = float(rival_lineup.get("interceptions_p90", pd.Series([1.2])).mean())
        else:
            away_off, away_def = 1.15, 1.2

        # Ataque local vs defensa visitante y ataque visitante vs defensa local
        lam = max(0.35, (home_off * (1.35 / max(away_def, 0.4))) + home_advantage)
        mu = max(0.25, (away_off * (1.35 / max(home_def, 0.4))))

        # Ponderación contextual de Elo si está disponible
        h_elo = home_elo if home_elo is not None else kwargs.get("elo_home")
        a_elo = away_elo if away_elo is not None else kwargs.get("elo_away")
        if h_elo is not None and a_elo is not None:
            elo_diff = (float(h_elo) - float(a_elo)) / 400.0
            lam = max(0.30, lam * (1.0 + elo_diff * 0.15))
            mu = max(0.20, mu * (1.0 - elo_diff * 0.15))

        from scipy.stats import poisson
        max_g = 7
        home_p = np.array([poisson.pmf(i, lam) for i in range(max_g)])
        away_p = np.array([poisson.pmf(j, mu) for j in range(max_g)])
        mat = np.outer(home_p, away_p)
        mat = mat / mat.sum()

        win_home = float(np.sum(np.tril(mat, -1)))
        draw = float(np.sum(np.diag(mat)))
        win_away = float(np.sum(np.triu(mat, 1)))

        tot_p = win_home + draw + win_away
        norm_h = win_home / tot_p
        norm_a = win_away / tot_p

        # Normalización estricta: la suma redondeada da exactamente 1.0000
        w_h = round(norm_h, 4)
        w_a = round(norm_a, 4)
        dr = round(1.0 - (w_h + w_a), 4)

        return {
            "win_home_prob": w_h,
            "draw_prob": dr,
            "win_away_prob": w_a,
            "lambda_home": round(lam, 2),
            "lambda_away": round(mu, 2),
            "home_team": home_team,
            "away_team": away_team,
        }
