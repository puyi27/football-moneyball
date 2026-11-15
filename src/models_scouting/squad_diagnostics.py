from typing import Optional, Any, Dict, Union, List
import types
import numpy as np
import pandas as pd
from src.data.global_player_database import load_global_players


def _deficit_items_wrapper(df_instance: pd.DataFrame):
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
    """
    Motor de Diagnóstico Integral de Plantillas y Alertas Deportivas / Financieras (Moneyball).
    Evalúa:
      1. Déficits por posición frente a la media de la competición.
      2. Riesgos de veteranía y sobrecarga de minutos por puesto.
      3. Eficiencia salarial y coste por punto de rendimiento por demarcación táctica.
      4. Coherencia multitemporada evitando la mezcla espuria de temporadas.
    """

    def __init__(self, df_metrics: Optional[pd.DataFrame] = None):
        """
        Inicializa el analizador de plantilla con el dataset de métricas procesadas.
        Si df_metrics es None, carga la base de datos global de referencia.
        """
        if df_metrics is not None and not df_metrics.empty:
            self.df = df_metrics.copy()
        else:
            self.df = load_global_players()

    def analyze_team_deficits(
        self,
        team_name: Any = "Sevilla FC",
        metric: str = "goals_p90",
        season: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compara el promedio de una métrica por posición en un equipo frente al promedio general de la liga.
        Acepta en team_name tanto un string con el nombre del club como un DataFrame de plantilla.
        Soporta filtrado multitemporada para garantizar mediciones interanuales rigurosas.
        """
        base_df = self.df.copy()

        # Resolver el nombre del club y la temporada del input
        if isinstance(team_name, pd.DataFrame):
            if "season" in team_name.columns and season is None and not team_name.empty:
                season = str(team_name["season"].iloc[0])
            if "team_name" in team_name.columns and not team_name.empty:
                t_name = str(team_name["team_name"].iloc[0])
            else:
                t_name = "Sevilla FC"
        else:
            t_name = str(team_name)

        # Filtrar base de datos por temporada para evitar mezcla de campañas
        if season and "season" in base_df.columns:
            s_df = base_df[base_df["season"] == season]
            if not s_df.empty:
                base_df = s_df
        elif "season" in base_df.columns and base_df["season"].nunique() > 1:
            # Por defecto, evaluar sobre la temporada más reciente
            latest_s = base_df["season"].max()
            base_df = base_df[base_df["season"] == latest_s]

        if t_name not in base_df["team_name"].values:
            # Búsqueda difusa de club
            matches = [t for t in base_df["team_name"].unique() if t_name.lower() in t.lower()]
            if matches:
                t_name = matches[0]
            elif not base_df.empty:
                t_name = base_df["team_name"].iloc[0]

        eval_metric = metric if metric in base_df.columns else ("xg_p90" if "xg_p90" in base_df.columns else "minutes")

        # Promedio por posición en LaLiga
        league_avg = base_df.groupby("position")[eval_metric].mean().reset_index()
        league_avg.rename(columns={eval_metric: "league_avg"}, inplace=True)

        # Promedio por posición en el equipo objetivo
        team_df = base_df[base_df["team_name"] == t_name]
        team_avg = team_df.groupby("position")[eval_metric].mean().reset_index()
        team_avg.rename(columns={eval_metric: "team_avg"}, inplace=True)

        # Unir y calcular la diferencia/déficit
        comparison = pd.merge(league_avg, team_avg, on="position", how="left").fillna(0.0)
        comparison["deficit"] = comparison["team_avg"] - comparison["league_avg"]

        # Identificar si hay déficit respecto al promedio de la liga
        comparison["needs_reinforcement"] = comparison["deficit"] < 0

        # Enriquecer con dictamen descriptivo y status
        def get_status_info(row):
            defic = float(row["deficit"])
            pos = str(row["position"])
            if defic < -0.15:
                return "CRITICAL", f"Déficit severo en {pos} ({defic:+.2f} {eval_metric} vs media de liga). Urge refuerzo prioritario en mercado."
            elif defic < 0.0:
                return "WARNING", f"Rendimiento por debajo de la media en {pos} ({defic:+.2f} {eval_metric}). Monitorear rotación."
            elif defic > 0.15:
                return "OK", f"Línea de {pos} de alto rendimiento (+{defic:.2f} sobre media de liga)."
            else:
                return "BALANCED", f"Línea de {pos} equilibrada (+{defic:.2f} en rango de competición)."

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
        min_minutes_share: float = 0.6,
        season: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detecta alertas de dependencia de veteranos (jugadores mayores de max_age 
        que acumulan un porcentaje alto de minutos en su posición).
        Soporta filtrado multitemporada para no acumular minutos de ejercicios pasados.
        """
        if isinstance(team_name, pd.DataFrame):
            team_df = team_name.copy()
            if season and "season" in team_df.columns:
                team_df = team_df[team_df["season"] == season]
        else:
            base_df = self.df.copy()
            if season and "season" in base_df.columns:
                base_df = base_df[base_df["season"] == season]
            elif "season" in base_df.columns and base_df["season"].nunique() > 1:
                base_df = base_df[base_df["season"] == base_df["season"].max()]

            t_name = str(team_name)
            if t_name not in base_df["team_name"].values:
                matches = [t for t in base_df["team_name"].unique() if t_name.lower() in t.lower()]
                t_name = matches[0] if matches else base_df["team_name"].iloc[0]
            team_df = base_df[base_df["team_name"] == t_name].copy()

        if team_df.empty or "minutes" not in team_df.columns or "age" not in team_df.columns:
            return pd.DataFrame(columns=["player_name", "position", "age", "minutes", "minutes_share"])

        # Calcular el total de minutos jugados por posición en el equipo
        total_mins_position = team_df.groupby("position")["minutes"].transform("sum")
        team_df["minutes_share"] = (team_df["minutes"] / total_mins_position.replace(0, 1)).round(3)

        # Filtrar alertas: veteranos con alta carga de minutos
        alerts = team_df[
            (team_df["age"] >= max_age) & 
            (team_df["minutes_share"] >= min_minutes_share)
        ].copy()

        cols = [c for c in ["player_name", "position", "age", "minutes", "minutes_share"] if c in alerts.columns]
        return alerts[cols].sort_values(by="minutes_share", ascending=False).reset_index(drop=True)

    def analyze_team_payroll_efficiency(
        self,
        team_name: Any = "Sevilla FC",
        season: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Cruza el rendimiento p90 con el valor de mercado y coste salarial por línea posicional
        para detectar líneas con sobrecoste financiero o alta rentabilidad Moneyball.
        """
        base_df = self.df.copy()
        if isinstance(team_name, pd.DataFrame):
            t_name = str(team_name["team_name"].iloc[0]) if "team_name" in team_name.columns and not team_name.empty else "Sevilla FC"
        else:
            t_name = str(team_name)

        if season and "season" in base_df.columns:
            base_df = base_df[base_df["season"] == season]
        elif "season" in base_df.columns and base_df["season"].nunique() > 1:
            base_df = base_df[base_df["season"] == base_df["season"].max()]

        team_df = base_df[base_df["team_name"] == t_name].copy()
        if team_df.empty:
            return pd.DataFrame()

        # Estimar salario y amortización por jugador
        team_df["mv"] = team_df["market_value_m"].fillna(5.0)
        team_df["estimated_wage_m"] = team_df["mv"] * 0.08
        team_df["annual_cost_m"] = team_df["estimated_wage_m"] + (team_df["mv"] / 4.0)

        # Rendimiento agregado
        team_df["impact_p90"] = (
            team_df["xg_p90"].fillna(0) + team_df["xa_p90"].fillna(0) +
            team_df["interceptions_p90"].fillna(0) * 0.5 + team_df["tackles_p90"].fillna(0) * 0.5
        )

        agg = team_df.groupby("position").agg(
            player_count=("player_id", "count"),
            avg_age=("age", "mean"),
            total_market_val_m=("mv", "sum"),
            total_annual_cost_m=("annual_cost_m", "sum"),
            avg_p90_impact=("impact_p90", "mean")
        ).reset_index()

        agg["avg_age"] = agg["avg_age"].round(1)
        agg["total_market_val_m"] = agg["total_market_val_m"].round(1)
        agg["total_annual_cost_m"] = agg["total_annual_cost_m"].round(1)
        agg["avg_p90_impact"] = agg["avg_p90_impact"].round(2)
        agg["cost_per_impact_point_m"] = (agg["total_annual_cost_m"] / agg["avg_p90_impact"].clip(lower=0.1)).round(2)

        return agg.sort_values(by="total_market_val_m", ascending=False).reset_index(drop=True)

