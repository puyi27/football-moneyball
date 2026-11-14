import numpy as np
import pandas as pd

class EloSystem:
    def __init__(self, k_factor: float = 20.0, home_advantage: float = 100.0, base_rating: float = 1500.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.base_rating = base_rating
        self.ratings = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.base_rating)

    def expected_result(self, rating_a: float, rating_b: float) -> float:
        """Calcula la probabilidad de victoria de A sobre B usando la curva logística."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, home_team: str, away_team: str, home_goals: int, away_goals: int):
        """Actualiza los ratings tras un partido, considerando la ventaja de campo."""
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

        # Ajustar por margen de victoria (opcional, refinamiento común en fútbol)
        goal_diff = abs(home_goals - away_goals)
        margin_multiplier = np.log(goal_diff + 1) if goal_diff > 0 else 1.0

        # Nuevos ratings (restamos el home_advantage temporal)
        new_home = self.get_rating(home_team) + self.k_factor * margin_multiplier * (s_home - e_home)
        new_away = self.get_rating(away_team) + self.k_factor * margin_multiplier * (s_away - e_away)

        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away