from typing import Optional, Any, Dict, Union
import types
import pandas as pd
from src.data.global_player_database import load_global_players


def _deficit_items_wrapper(df_instance):
    """Genera la estructura de iteración clave-valor requerida por el Streamlit dashboard (app.py)."""
    result = {}
    for _, row in df_instance.iterrows():
        pos = str(row.get("position", "Demarcación"))
        result[pos] = {
            "status": str(row.get("status", "OK")),
            "reason": str(row.get("reason", "")),
            "deficit": float(row.get("deficit", 0.0)),
            "needs_reinforcement": bool(row.get("needs_reinforcement", False)),
            "team_avg": float(row.get("team_avg", 0.0)),
            "league_avg": float(row.get("league_avg", 0.0)),
        }
    return result.items()


class SquadDiagnostics:
    def __init__(self, df_metrics: Optional[pd.DataFrame] = None):
        """
        Inicializa el analizador de plantilla con el dataset de métricas procesadas.
        Si df_metrics es None, carga la base de datos global de referencia.
        """
        if df_metrics is not None and not df_metrics.empty:
            self.df = df_metrics.copy()
        else:
            self.df = load_global_players()

    def analyze_team_deficits(self, team_name: Any = "Sevilla FC", metric: str = "goals_p90") -> pd.DataFrame:
        """
        Compara el promedio de una métrica por posición en un equipo frente al promedio general de la liga.
        Acepta en team_name tanto un string con el nombre del club como un DataFrame de plantilla.
        """
        if isinstance(team_name, pd.DataFrame):
            if "team_name" in team_name.columns and not team_name.empty:
                t_name = str(team_name["team_name"].iloc[0])
            else:
                t_name = "Sevilla FC"
        else:
            t_name = str(team_name)

        if t_name not in self.df["team_name"].values:
            # Si no coincide exactamente, buscar coincidencia parcial o usar el primer club
            matches = [t for t in self.df["team_name"].unique() if t_name.lower() in t.lower()]
            if matches:
                t_name = matches[0]
            elif not self.df.empty:
                t_name = self.df["team_name"].iloc[0]

        eval_metric = metric if metric in self.df.columns else ("xg_p90" if "xg_p90" in self.df.columns else "minutes")

        # Promedio por posición en LaLiga
        league_avg = self.df.groupby("position")[eval_metric].mean().reset_index()
        league_avg.rename(columns={eval_metric: "league_avg"}, inplace=True)

        # Promedio por posición en el equipo objetivo
        team_df = self.df[self.df["team_name"] == t_name]
        team_avg = team_df.groupby("position")[eval_metric].mean().reset_index()
        team_avg.rename(columns={eval_metric: "team_avg"}, inplace=True)

        # Unir y calcular la diferencia/déficit
        comparison = pd.merge(league_avg, team_avg, on="position", how="left").fillna(0.0)
        comparison["deficit"] = comparison["team_avg"] - comparison["league_avg"]

        # Identificar si hay déficit respecto al promedio de la liga
        comparison["needs_reinforcement"] = comparison["deficit"] < 0

        # Enriquecer con dictamen descriptivo y status
        def get_status_info(row):
            defic = row["deficit"]
            pos = row["position"]
            if defic < -0.15:
                return "CRITICAL", f"Déficit severo en {pos} ({defic:+.2f} {eval_metric} vs media de liga). Urge refuerzo prioritario en mercado."
            elif defic < 0.0:
                return "WARNING", f"Rendimiento por debajo de la media en {pos} ({defic:+.2f} {eval_metric}). Monitorear rotación."
            else:
                return "OK", f"Línea de {pos} solvente (+{defic:.2f} sobre media de liga)."

        statuses = [get_status_info(r) for _, r in comparison.iterrows()]
        comparison["status"] = [s[0] for s in statuses]
        comparison["reason"] = [s[1] for s in statuses]

        sorted_df = comparison.sort_values(by="deficit", ascending=True).reset_index(drop=True)
        sorted_df.items = types.MethodType(lambda self: _deficit_items_wrapper(self), sorted_df)
        return sorted_df

    def detect_squad_risk_alerts(
        self,
        team_name: Any = "Sevilla FC",
        max_age: int = 30,
        min_minutes_share: float = 0.6
    ) -> pd.DataFrame:
        """
        Detecta alertas de dependencia de veteranos (jugadores mayores de max_age 
        que acumulan un porcentaje alto de minutos en su posición).
        """
        if isinstance(team_name, pd.DataFrame):
            team_df = team_name.copy()
        else:
            t_name = str(team_name)
            if t_name not in self.df["team_name"].values:
                matches = [t for t in self.df["team_name"].unique() if t_name.lower() in t.lower()]
                t_name = matches[0] if matches else self.df["team_name"].iloc[0]
            team_df = self.df[self.df["team_name"] == t_name].copy()

        if team_df.empty or "minutes" not in team_df.columns or "age" not in team_df.columns:
            return pd.DataFrame(columns=["player_name", "position", "age", "minutes", "minutes_share"])

        # Calcular el total de minutos jugados por posición en el equipo
        total_mins_position = team_df.groupby("position")["minutes"].transform("sum")
        team_df["minutes_share"] = team_df["minutes"] / total_mins_position.replace(0, 1)

        # Filtrar alertas: veteranos con alta carga de minutos
        alerts = team_df[
            (team_df["age"] >= max_age) & 
            (team_df["minutes_share"] >= min_minutes_share)
        ]

        cols = [c for c in ["player_name", "position", "age", "minutes", "minutes_share"] if c in alerts.columns]
        return alerts[cols].copy()
