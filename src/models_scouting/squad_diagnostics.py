import pandas as pd

class SquadDiagnostics:
    def __init__(self, df_metrics: pd.DataFrame):
        """
        Inicializa el analizador de plantilla con el dataset de métricas procesadas.
        """
        self.df = df_metrics.copy()

    def analyze_team_deficits(self, team_name: str, metric: str = 'goals_p90') -> pd.DataFrame:
        """
        Compara el promedio de una métrica por posición en un equipo frente al promedio general de la liga.
        """
        if team_name not in self.df['team_name'].values:
            raise ValueError(f"El equipo '{team_name}' no existe en el conjunto de datos.")

        # Promedio por posición en LaLiga
        league_avg = self.df.groupby('position')[metric].mean().reset_index()
        league_avg.rename(columns={metric: 'league_avg'}, inplace=True)

        # Promedio por posición en el equipo objetivo
        team_df = self.df[self.df['team_name'] == team_name]
        team_avg = team_df.groupby('position')[metric].mean().reset_index()
        team_avg.rename(columns={metric: 'team_avg'}, inplace=True)

        # Unir y calcular la diferencia/déficit
        comparison = pd.merge(league_avg, team_avg, on='position', how='left').fillna(0.0)
        comparison['deficit'] = comparison['team_avg'] - comparison['league_avg']
        
        # Identificar si hay déficit respecto al promedio de la liga
        comparison['needs_reinforcement'] = comparison['deficit'] < 0

        return comparison.sort_values(by='deficit', ascending=True)

    def detect_squad_risk_alerts(self, team_name: str, max_age: int = 30, min_minutes_share: float = 0.6) -> pd.DataFrame:
        """
        Detecta alertas de dependencia de veteranos (jugadores mayores de max_age 
        que acumulan un porcentaje alto de minutos en su posición).
        """
        team_df = self.df[self.df['team_name'] == team_name].copy()
        if team_df.empty or 'minutes' not in team_df.columns or 'age' not in team_df.columns:
            return pd.DataFrame()

        # Calcular el total de minutos jugados por posición en el equipo
        total_mins_position = team_df.groupby('position')['minutes'].transform('sum')
        team_df['minutes_share'] = team_df['minutes'] / total_mins_position.replace(0, 1)

        # Filtrar alertas: veteranos con alta carga de minutos
        alerts = team_df[
            (team_df['age'] >= max_age) & 
            (team_df['minutes_share'] >= min_minutes_share)
        ]

        return alerts[['player_name', 'position', 'age', 'minutes', 'minutes_share']]