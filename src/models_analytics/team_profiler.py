from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from src.models_analytics.elo_model import EloSystem


class TeamProfiler:
    """
    Motor de Análisis y Perfilado Táctico de Equipos.
    Calcula el ADN táctico del club (estilo de juego, posesión, presión, verticalidad),
    evalúa la profundidad de plantilla y analiza la pirámide de edad
    para la planificación deportiva del director técnico.
    Totalmente desacoplado de la persistencia directa: opera sobre DataFrames.
    """

    # Perfiles tácticos de referencia por club
    TEAM_TACTICAL_STYLES = {
        "Real Madrid": {"possession_pct": 59.5, "pressing_intensity": 7.2, "verticality": 7.8, "style": "Transición Rápida y Pegada Letal", "preferred_channel": "Banda Izquierda (Vinícius/Mbappé)"},
        "Barcelona": {"possession_pct": 65.0, "pressing_intensity": 8.8, "verticality": 5.4, "style": "Bloque Alto, Posesión y Presión Asfixiante", "preferred_channel": "Bandas Abiertas (Yamal/Raphinha)"},
        "Atletico Madrid": {"possession_pct": 51.5, "pressing_intensity": 8.1, "verticality": 7.1, "style": "Bloque Medio Sólido y Salida Vertical", "preferred_channel": "Carril Central y Segunda Línea (Griezmann)"},
        "Athletic Club": {"possession_pct": 52.0, "pressing_intensity": 8.5, "verticality": 8.4, "style": "Ritmo Alto, Acoso Físico y Desborde Exterior", "preferred_channel": "Banda Izquierda (Nico Williams)"},
        "Real Sociedad": {"possession_pct": 56.5, "pressing_intensity": 7.8, "verticality": 5.8, "style": "Construcción Paciente desde Pivote y Técnica", "preferred_channel": "Asociación Interior (Zubimendi/Brais)"},
        "Real Betis": {"possession_pct": 54.0, "pressing_intensity": 6.8, "verticality": 5.5, "style": "Fantasía Interior, Juego al Pie y Pausa", "preferred_channel": "Pasillos Interiores (Isco/Lo Celso)"},
        "Sevilla FC": {"possession_pct": 49.0, "pressing_intensity": 7.1, "verticality": 6.8, "style": "Juego Directo, Disputa en Segunda Jugada y Transición", "preferred_channel": "Centros al Área e Incursión Exterior"},
        "Girona FC": {"possession_pct": 57.0, "pressing_intensity": 7.5, "verticality": 6.5, "style": "Salida Limpia desde Atrás y Laterales Invertidos", "preferred_channel": "Sobrecarga Interior (Gutiérrez)"},
        "Villarreal CF": {"possession_pct": 53.0, "pressing_intensity": 6.9, "verticality": 7.2, "style": "Ataque Combinativo Rápido y Desmarque de Ruptura", "preferred_channel": "Último Tercio y Filtración (Baena)"},
        "Manchester City": {"possession_pct": 66.5, "pressing_intensity": 8.2, "verticality": 4.8, "style": "Dominio Posicional Absoluto y Ataque por Desgaste", "preferred_channel": "Control Total de Pasillo Central (Rodri)"},
        "Bayern Munich": {"possession_pct": 63.0, "pressing_intensity": 8.6, "verticality": 7.5, "style": "Presión Alta Feroz y Despliegue Ofensivo Masivo", "preferred_channel": "Espacio Entre Líneas (Musiala/Kane)"},
        "Inter": {"possession_pct": 55.0, "pressing_intensity": 7.6, "verticality": 7.0, "style": "Estructura 3-5-2 Elástica y Salidas en Bloque", "preferred_channel": "Transición Guiada por Medios (Barella)"},
        "Sporting CP": {"possession_pct": 61.0, "pressing_intensity": 8.4, "verticality": 7.9, "style": "Presión Tras Pérdida y Verticalidad Hacia Delantero Referencia", "preferred_channel": "Espacio para Gyökeres"},
    }

    def __init__(self, df_players: Optional[pd.DataFrame] = None, elo_system: Optional[EloSystem] = None):
        if df_players is not None and not df_players.empty:
            self.df = df_players.copy()
        else:
            try:
                from src.data.global_player_database import load_global_players
                self.df = load_global_players()
            except Exception:
                self.df = pd.DataFrame()
        self.elo_system = elo_system or EloSystem()

    def get_available_teams(self) -> List[str]:
        """Devuelve la lista ordenada de equipos disponibles en el censo."""
        if self.df.empty or "team_name" not in self.df.columns:
            return list(self.TEAM_TACTICAL_STYLES.keys())
        return sorted([str(t) for t in self.df["team_name"].dropna().unique()])

    def get_team_overview(self, team_name: str) -> Dict[str, Any]:
        """Calcula las métricas y perfil táctico/deportivo completo de un equipo."""
        if self.df.empty or "team_name" not in self.df.columns:
            raise ValueError(f"No hay datos de plantilla disponibles para perfilar '{team_name}'.")

        team_players = self.df[self.df["team_name"] == team_name]
        if team_players.empty:
            raise ValueError(f"El equipo '{team_name}' no existe en la base de datos.")

        # Elo rating
        current_elo = self.elo_system.get_rating(team_name)

        # Valores agregados seguros contra NaNs
        total_market_value = float(np.nan_to_num(team_players.get("market_value_m", pd.Series([0.0])).sum(), nan=0.0))
        
        age_series = team_players.get("age", pd.Series([26.0])).dropna()
        avg_age = float(age_series.mean()) if not age_series.empty else 26.0

        total_minutes = int(np.nan_to_num(team_players.get("minutes", pd.Series([0])).sum(), nan=0))

        # Poder ofensivo estimado (alpha) y defensivo (beta) robustos
        pos_col = team_players.get("position", pd.Series([""] * len(team_players)))
        fw_players = team_players[pos_col == "FW"]
        if not fw_players.empty and "xg_p90" in fw_players.columns and not fw_players["xg_p90"].dropna().empty:
            top_fw_xg = float(fw_players["xg_p90"].dropna().mean())
        else:
            top_fw_xg = 0.35
        alpha_est = round(float(np.clip(0.8 + (top_fw_xg * 1.2), 0.7, 1.8)), 2)

        df_players = team_players[pos_col == "DF"]
        if not df_players.empty and "interceptions_p90" in df_players.columns and not df_players["interceptions_p90"].dropna().empty:
            top_df_int = float(df_players["interceptions_p90"].dropna().mean())
        else:
            top_df_int = 1.5
        beta_est = round(float(np.clip(1.6 - (top_df_int * 0.35), 0.6, 1.5)), 2)

        # Parámetros tácticos (de referencia o inferidos dinámicamente)
        if team_name in self.TEAM_TACTICAL_STYLES:
            tactics = self.TEAM_TACTICAL_STYLES[team_name]
        else:
            possession = float(np.clip(50.0 + (alpha_est - 1.0) * 10.0, 42.0, 65.0))
            pressing = float(np.clip(7.0 + (1.2 - beta_est) * 3.0, 5.0, 9.0))
            verticality = float(np.clip(7.0 - (possession - 50.0) * 0.1, 4.5, 8.5))
            style = "Posesión y Control" if possession >= 56 else ("Transición Rápida" if verticality >= 7.2 else "Equilibrado")
            tactics = {
                "possession_pct": round(possession, 1),
                "pressing_intensity": round(pressing, 1),
                "verticality": round(verticality, 1),
                "style": style,
                "preferred_channel": "Carril Central"
            }

        # Detección de puestos que necesitan relevo urgente (veteranos >=31 años con muchos minutos)
        aging_positions = []
        if "age" in team_players.columns and "minutes" in team_players.columns:
            veterans = team_players[(team_players["age"] >= 31) & (team_players["minutes"] >= 1400)]
            for _, vet in veterans.iterrows():
                p_name = vet.get("player_name", "Veterano")
                pos = vet.get("position", "N/A")
                age_val = int(vet.get("age", 31))
                min_val = int(vet.get("minutes", 1400))
                aging_positions.append({
                    "player_name": p_name,
                    "position": pos,
                    "age": age_val,
                    "minutes": min_val,
                    "urgency": "Alta" if age_val >= 33 else "Media"
                })

        # Conteo de canteranos en el primer equipo
        acad_count = int((team_players["is_academy"] == True).sum()) if "is_academy" in team_players.columns else 0

        # Roster ordenado
        roster = team_players.sort_values(by="minutes", ascending=False) if "minutes" in team_players.columns else team_players

        return {
            "team_name": team_name,
            "league": team_players.iloc[0].get("league", "LaLiga") if not team_players.empty else "LaLiga",
            "elo_rating": round(float(current_elo), 1),
            "alpha_offensive_power": alpha_est,
            "beta_defensive_leak": beta_est,
            "total_market_value_m": round(total_market_value, 1),
            "avg_squad_age": round(avg_age, 1),
            "avg_age": round(avg_age, 1),
            "total_squad_size": len(team_players),
            "squad_size": len(team_players),
            "total_minutes": total_minutes,
            "academy_players_count": acad_count,
            "tactical_adn": tactics,
            "aging_renewals_needed": aging_positions,
            "squad_roster": roster
        }

    def compare_teams(self, team_a: str, team_b: str) -> Dict[str, Any]:
        """Compara táctica, potencial ofensivo/defensivo y rating Elo entre dos equipos."""
        overview_a = self.get_team_overview(team_a)
        overview_b = self.get_team_overview(team_b)

        elo_diff = overview_a["elo_rating"] - overview_b["elo_rating"]
        e_a = self.elo_system.expected_result(overview_a["elo_rating"], overview_b["elo_rating"])

        return {
            "team_a": team_a,
            "team_b": team_b,
            "elo_diff": round(elo_diff, 1),
            "win_expectation_a": round(e_a, 3),
            "win_expectation_b": round(1.0 - e_a, 3),
            "alpha_comparison": {team_a: overview_a["alpha_offensive_power"], team_b: overview_b["alpha_offensive_power"]},
            "beta_comparison": {team_a: overview_a["beta_defensive_leak"], team_b: overview_b["beta_defensive_leak"]},
            "tactics_a": overview_a["tactical_adn"],
            "tactics_b": overview_b["tactical_adn"],
        }

