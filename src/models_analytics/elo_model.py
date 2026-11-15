from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd


class EloSystem:
    """
    Motor matemático puro de puntuación Elo Dinámico para fútbol profesional.
    Implementa la formulación oficial de World Football Elo Ratings con ajuste
    por ventaja de localía, multiplicador por margen de victoria e identificabilidad de escala.
    Totalmente desacoplado de la base de datos y de la capa de interfaz.
    """

    def __init__(self, k_factor: float = 20.0, home_advantage: float = 100.0, base_rating: float = 1500.0):
        self.k_factor = float(k_factor)
        self.home_advantage = float(home_advantage)
        self.base_rating = float(base_rating)
        self.ratings: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Obtiene el rating actual del equipo o el valor base si no ha sido registrado."""
        return float(self.ratings.get(team, self.base_rating))

    def expected_result(self, rating_a: float, rating_b: float) -> float:
        """
        Calcula la expectativa de resultado de A sobre B mediante la curva logística.
        E_A = 1 / (1 + 10^((R_B - R_A) / 400)).
        """
        diff = np.clip((rating_b - rating_a) / 400.0, -10.0, 10.0)
        return float(1.0 / (1.0 + 10.0 ** diff))

    def calculate_margin_multiplier(self, home_goals: int, away_goals: int) -> float:
        """
        Multiplicador por margen de victoria (World Football Elo Ratings standard):
        - Empates o victorias por 1 gol: N=1.0
        - Victorias por 2 goles: N=1.5
        - Victorias por >=3 goles: (11 + N) / 8.0
        """
        goal_diff = abs(int(home_goals) - int(away_goals))
        if goal_diff <= 1:
            return 1.0
        elif goal_diff == 2:
            return 1.5
        else:
            return float((11.0 + goal_diff) / 8.0)

    def update_ratings(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> Tuple[float, float]:
        """
        Actualiza los ratings tras un partido considerando la ventaja de campo.
        Conserva estrictamente la propiedad de juego de suma cero en expectativas.
        """
        home_rating = self.get_rating(home_team) + self.home_advantage
        away_rating = self.get_rating(away_team)

        # Determinar el resultado (1=Victoria Local, 0.5=Empate, 0=Victoria Visitante)
        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals == away_goals:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0

        # Calcular expectativas
        e_home = self.expected_result(home_rating, away_rating)
        e_away = self.expected_result(away_rating, home_rating)

        # Ajustar por margen de victoria
        margin_multiplier = self.calculate_margin_multiplier(home_goals, away_goals)

        # Nuevos ratings (restamos el home_advantage temporal)
        delta_home = self.k_factor * margin_multiplier * (s_home - e_home)
        delta_away = self.k_factor * margin_multiplier * (s_away - e_away)

        new_home = self.get_rating(home_team) + delta_home
        new_away = self.get_rating(away_team) + delta_away

        self.ratings[home_team] = float(new_home)
        self.ratings[away_team] = float(new_away)
        return float(new_home), float(new_away)

    def predict_match_probabilities(self, home_team: str, away_team: str, neutral_venue: bool = False) -> Dict[str, float]:
        """
        Estima probabilidades de resultado 1X2 mediante calibración logística de la diferencia Elo.
        Devuelve {'home_win': float, 'draw': float, 'away_win': float} sumando 1.0.
        """
        home_adv = 0.0 if neutral_venue else self.home_advantage
        r_h = self.get_rating(home_team) + home_adv
        r_a = self.get_rating(away_team)

        e_h = self.expected_result(r_h, r_a)
        e_a = 1.0 - e_h

        diff = abs(r_h - r_a)
        draw_prob = float(np.clip(0.27 * np.exp(-diff / 600.0), 0.08, 0.32))

        win_h = float(np.clip(e_h - 0.5 * draw_prob, 0.02, 0.95))
        win_a = float(np.clip(e_a - 0.5 * draw_prob, 0.02, 0.95))

        total = win_h + draw_prob + win_a
        return {
            "home_win": round(win_h / total, 4),
            "draw": round(draw_prob / total, 4),
            "away_win": round(win_a / total, 4)
        }

    def batch_update(self, df_matches: pd.DataFrame) -> Dict[str, float]:
        """Procesa una secuencia de partidos cronológicos y actualiza los ratings."""
        for _, row in df_matches.iterrows():
            self.update_ratings(
                str(row['home_team']),
                str(row['away_team']),
                int(row['home_goals']),
                int(row['away_goals'])
            )
        return self.ratings

    def copy(self) -> 'EloSystem':
        """Devuelve una copia independiente para simulaciones y ramificaciones de escenarios."""
        clone = EloSystem(
            k_factor=self.k_factor,
            home_advantage=self.home_advantage,
            base_rating=self.base_rating
        )
        clone.ratings = self.ratings.copy()
        return clone

    def reset(self) -> None:
        """Reinicia todos los ratings acumulados al estado inicial."""
        self.ratings.clear()
