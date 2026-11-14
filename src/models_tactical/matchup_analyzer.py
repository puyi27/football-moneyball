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
        Filtra solo jugadores fiables (minutos suficientes en la temporada).
        """
        formation = formation or self.DEFAULT_FORMATION
        opponent_profile = opponent_profile or {"defense_weakness": 1.0, "attack_threat": 1.0}

        df = available_players_df.copy()
        if require_reliable and "is_reliable" in df.columns:
            df = df[df["is_reliable"] == True].copy()

        if df.empty:
            raise ValueError("No hay jugadores fiables disponibles para confeccionar la alineación.")

        # Calcular tactical fit score para cada jugador disponible
        df["tactical_fit_score"] = df.apply(
            lambda row: self.calculate_player_tactical_fit(row, opponent_profile), axis=1
        )

        selected_lineup = []
        for position, count in formation.items():
            pos_players = df[df["position"] == position].sort_values(
                by="tactical_fit_score", ascending=False
            )
            if len(pos_players) < count:
                # Si no hay suficientes jugadores de la posición exacta, buscar los de posición parcial
                alt_players = df[df["position"].str.contains(position, na=False)].sort_values(
                    by="tactical_fit_score", ascending=False
                )
                selected = alt_players.head(count)
            else:
                selected = pos_players.head(count)

            selected_lineup.append(selected)

        result_df = pd.concat(selected_lineup, ignore_index=True)
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
