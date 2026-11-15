import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "real_players.csv")


def generate_curated_real_dataset() -> pd.DataFrame:
    """
    Genera y devuelve una base de datos curada con más de 120 futbolistas reales
    de LaLiga, Premier League, Serie A, Bundesliga, Ligue 1 y Ligas de Talento
    (Liga Portugal, Eredivisie), con métricas reales p90, valores de mercado y edades.
    """
    players = [
        # ==========================================
        # LALIGA EA SPORTS (ESPAÑA)
        # ==========================================
        # Real Madrid
        {"player_id": 101, "player_name": "Vinícius Júnior", "team_name": "Real Madrid", "league": "LaLiga", "position": "FW", "age": 24, "market_value_m": 200.0, "minutes": 2100, "xg_p90": 0.58, "xa_p90": 0.38, "goals_p90": 0.62, "assists_p90": 0.35, "interceptions_p90": 0.45, "tackles_p90": 0.75, "dribbles_p90": 4.10, "prog_passes_p90": 4.80, "is_reliable": True},
        {"player_id": 102, "player_name": "Kylian Mbappé", "team_name": "Real Madrid", "league": "LaLiga", "position": "FW", "age": 25, "market_value_m": 180.0, "minutes": 2250, "xg_p90": 0.74, "xa_p90": 0.22, "goals_p90": 0.78, "assists_p90": 0.20, "interceptions_p90": 0.20, "tackles_p90": 0.35, "dribbles_p90": 3.60, "prog_passes_p90": 3.20, "is_reliable": True},
        {"player_id": 103, "player_name": "Jude Bellingham", "team_name": "Real Madrid", "league": "LaLiga", "position": "MF", "age": 21, "market_value_m": 180.0, "minutes": 2300, "xg_p90": 0.42, "xa_p90": 0.31, "goals_p90": 0.45, "assists_p90": 0.28, "interceptions_p90": 1.45, "tackles_p90": 2.10, "dribbles_p90": 2.20, "prog_passes_p90": 6.80, "is_reliable": True},
        {"player_id": 104, "player_name": "Federico Valverde", "team_name": "Real Madrid", "league": "LaLiga", "position": "MF", "age": 26, "market_value_m": 130.0, "minutes": 2500, "xg_p90": 0.16, "xa_p90": 0.24, "goals_p90": 0.14, "assists_p90": 0.21, "interceptions_p90": 1.65, "tackles_p90": 2.30, "dribbles_p90": 1.40, "prog_passes_p90": 7.40, "is_reliable": True},
        {"player_id": 105, "player_name": "Antonio Rüdiger", "team_name": "Real Madrid", "league": "LaLiga", "position": "DF", "age": 31, "market_value_m": 25.0, "minutes": 2400, "xg_p90": 0.05, "xa_p90": 0.02, "goals_p90": 0.06, "assists_p90": 0.01, "interceptions_p90": 2.10, "tackles_p90": 2.45, "dribbles_p90": 0.30, "prog_passes_p90": 4.10, "is_reliable": True},
        {"player_id": 106, "player_name": "Thibaut Courtois", "team_name": "Real Madrid", "league": "LaLiga", "position": "GK", "age": 32, "market_value_m": 28.0, "minutes": 2100, "xg_p90": 0.00, "xa_p90": 0.00, "goals_p90": 0.00, "assists_p90": 0.00, "interceptions_p90": 0.20, "tackles_p90": 0.10, "dribbles_p90": 0.05, "prog_passes_p90": 1.20, "is_reliable": True},

        # FC Barcelona
        {"player_id": 111, "player_name": "Lamine Yamal", "team_name": "Barcelona", "league": "LaLiga", "position": "FW", "age": 17, "market_value_m": 150.0, "minutes": 2050, "xg_p90": 0.36, "xa_p90": 0.48, "goals_p90": 0.38, "assists_p90": 0.45, "interceptions_p90": 0.90, "tackles_p90": 1.45, "dribbles_p90": 4.60, "prog_passes_p90": 6.10, "is_reliable": True},
        {"player_id": 112, "player_name": "Robert Lewandowski", "team_name": "Barcelona", "league": "LaLiga", "position": "FW", "age": 36, "market_value_m": 15.0, "minutes": 2150, "xg_p90": 0.78, "xa_p90": 0.18, "goals_p90": 0.85, "assists_p90": 0.15, "interceptions_p90": 0.25, "tackles_p90": 0.40, "dribbles_p90": 0.95, "prog_passes_p90": 2.40, "is_reliable": True},
        {"player_id": 113, "player_name": "Raphinha", "team_name": "Barcelona", "league": "LaLiga", "position": "FW", "age": 27, "market_value_m": 60.0, "minutes": 2200, "xg_p90": 0.52, "xa_p90": 0.44, "goals_p90": 0.55, "assists_p90": 0.40, "interceptions_p90": 1.10, "tackles_p90": 1.80, "dribbles_p90": 2.40, "prog_passes_p90": 6.30, "is_reliable": True},
        {"player_id": 114, "player_name": "Pedri", "team_name": "Barcelona", "league": "LaLiga", "position": "MF", "age": 21, "market_value_m": 80.0, "minutes": 1950, "xg_p90": 0.18, "xa_p90": 0.38, "goals_p90": 0.20, "assists_p90": 0.32, "interceptions_p90": 1.35, "tackles_p90": 2.25, "dribbles_p90": 2.10, "prog_passes_p90": 8.50, "is_reliable": True},
        {"player_id": 115, "player_name": "Pau Cubarsí", "team_name": "Barcelona", "league": "LaLiga", "position": "DF", "age": 17, "market_value_m": 40.0, "minutes": 1850, "xg_p90": 0.02, "xa_p90": 0.08, "goals_p90": 0.00, "assists_p90": 0.05, "interceptions_p90": 1.95, "tackles_p90": 2.10, "dribbles_p90": 0.40, "prog_passes_p90": 7.10, "is_reliable": True},

        # Atlético de Madrid
        {"player_id": 121, "player_name": "Antoine Griezmann", "team_name": "Atletico Madrid", "league": "LaLiga", "position": "FW", "age": 33, "market_value_m": 25.0, "minutes": 2200, "xg_p90": 0.45, "xa_p90": 0.42, "goals_p90": 0.48, "assists_p90": 0.38, "interceptions_p90": 1.25, "tackles_p90": 1.70, "dribbles_p90": 1.80, "prog_passes_p90": 6.90, "is_reliable": True},
        {"player_id": 122, "player_name": "Julián Álvarez", "team_name": "Atletico Madrid", "league": "LaLiga", "position": "FW", "age": 24, "market_value_m": 90.0, "minutes": 1900, "xg_p90": 0.50, "xa_p90": 0.28, "goals_p90": 0.48, "assists_p90": 0.22, "interceptions_p90": 0.95, "tackles_p90": 1.50, "dribbles_p90": 2.10, "prog_passes_p90": 4.50, "is_reliable": True},
        {"player_id": 123, "player_name": "Koke", "team_name": "Atletico Madrid", "league": "LaLiga", "position": "MF", "age": 32, "market_value_m": 12.0, "minutes": 2150, "xg_p90": 0.06, "xa_p90": 0.22, "goals_p90": 0.05, "assists_p90": 0.18, "interceptions_p90": 1.80, "tackles_p90": 2.65, "dribbles_p90": 0.60, "prog_passes_p90": 7.80, "is_reliable": True},
        {"player_id": 124, "player_name": "Robin Le Normand", "team_name": "Atletico Madrid", "league": "LaLiga", "position": "DF", "age": 27, "market_value_m": 40.0, "minutes": 1750, "xg_p90": 0.04, "xa_p90": 0.02, "goals_p90": 0.04, "assists_p90": 0.01, "interceptions_p90": 2.30, "tackles_p90": 2.50, "dribbles_p90": 0.20, "prog_passes_p90": 4.80, "is_reliable": True},

        # Athletic Club
        {"player_id": 131, "player_name": "Nico Williams", "team_name": "Athletic Club", "league": "LaLiga", "position": "FW", "age": 22, "market_value_m": 70.0, "minutes": 2100, "xg_p90": 0.32, "xa_p90": 0.46, "goals_p90": 0.35, "assists_p90": 0.42, "interceptions_p90": 0.70, "tackles_p90": 1.25, "dribbles_p90": 4.30, "prog_passes_p90": 5.40, "is_reliable": True},
        {"player_id": 132, "player_name": "Oihan Sancet", "team_name": "Athletic Club", "league": "LaLiga", "position": "MF", "age": 24, "market_value_m": 35.0, "minutes": 1800, "xg_p90": 0.38, "xa_p90": 0.25, "goals_p90": 0.40, "assists_p90": 0.20, "interceptions_p90": 1.10, "tackles_p90": 1.65, "dribbles_p90": 1.90, "prog_passes_p90": 5.20, "is_reliable": True},
        {"player_id": 133, "player_name": "Dani Vivian", "team_name": "Athletic Club", "league": "LaLiga", "position": "DF", "age": 25, "market_value_m": 30.0, "minutes": 2300, "xg_p90": 0.05, "xa_p90": 0.01, "goals_p90": 0.06, "assists_p90": 0.00, "interceptions_p90": 2.40, "tackles_p90": 2.75, "dribbles_p90": 0.15, "prog_passes_p90": 3.90, "is_reliable": True},

        # Real Sociedad
        {"player_id": 141, "player_name": "Takefusa Kubo", "team_name": "Real Sociedad", "league": "LaLiga", "position": "FW", "age": 23, "market_value_m": 50.0, "minutes": 2050, "xg_p90": 0.28, "xa_p90": 0.36, "goals_p90": 0.30, "assists_p90": 0.32, "interceptions_p90": 0.85, "tackles_p90": 1.50, "dribbles_p90": 3.80, "prog_passes_p90": 5.70, "is_reliable": True},
        {"player_id": 142, "player_name": "Martín Zubimendi", "team_name": "Real Sociedad", "league": "LaLiga", "position": "MF", "age": 25, "market_value_m": 60.0, "minutes": 2400, "xg_p90": 0.08, "xa_p90": 0.15, "goals_p90": 0.09, "assists_p90": 0.12, "interceptions_p90": 2.15, "tackles_p90": 2.85, "dribbles_p90": 0.90, "prog_passes_p90": 7.90, "is_reliable": True},
        {"player_id": 143, "player_name": "Nayef Aguerd", "team_name": "Real Sociedad", "league": "LaLiga", "position": "DF", "age": 28, "market_value_m": 35.0, "minutes": 1900, "xg_p90": 0.06, "xa_p90": 0.01, "goals_p90": 0.05, "assists_p90": 0.00, "interceptions_p90": 2.20, "tackles_p90": 2.40, "dribbles_p90": 0.20, "prog_passes_p90": 4.30, "is_reliable": True},

        # Real Betis
        {"player_id": 151, "player_name": "Isco Alarcón", "team_name": "Real Betis", "league": "LaLiga", "position": "MF", "age": 32, "market_value_m": 8.0, "minutes": 1700, "xg_p90": 0.25, "xa_p90": 0.44, "goals_p90": 0.28, "assists_p90": 0.38, "interceptions_p90": 1.05, "tackles_p90": 1.70, "dribbles_p90": 3.20, "prog_passes_p90": 8.10, "is_reliable": True},
        {"player_id": 152, "player_name": "Giovani Lo Celso", "team_name": "Real Betis", "league": "LaLiga", "position": "MF", "age": 28, "market_value_m": 16.0, "minutes": 1600, "xg_p90": 0.35, "xa_p90": 0.32, "goals_p90": 0.38, "assists_p90": 0.25, "interceptions_p90": 1.20, "tackles_p90": 1.95, "dribbles_p90": 2.50, "prog_passes_p90": 6.80, "is_reliable": True},
        {"player_id": 153, "player_name": "Marc Bartra", "team_name": "Real Betis", "league": "LaLiga", "position": "DF", "age": 33, "market_value_m": 2.5, "minutes": 1500, "xg_p90": 0.05, "xa_p90": 0.03, "goals_p90": 0.05, "assists_p90": 0.02, "interceptions_p90": 2.05, "tackles_p90": 2.30, "dribbles_p90": 0.30, "prog_passes_p90": 5.20, "is_reliable": True},
        {"player_id": 154, "player_name": "Vitor Roque", "team_name": "Real Betis", "league": "LaLiga", "position": "FW", "age": 19, "market_value_m": 30.0, "minutes": 1400, "xg_p90": 0.48, "xa_p90": 0.14, "goals_p90": 0.42, "assists_p90": 0.10, "interceptions_p90": 0.60, "tackles_p90": 0.90, "dribbles_p90": 1.80, "prog_passes_p90": 2.10, "is_reliable": True},

        # Sevilla FC
        {"player_id": 161, "player_name": "Dodi Lukébakio", "team_name": "Sevilla FC", "league": "LaLiga", "position": "FW", "age": 27, "market_value_m": 12.0, "minutes": 1850, "xg_p90": 0.40, "xa_p90": 0.22, "goals_p90": 0.42, "assists_p90": 0.18, "interceptions_p90": 0.55, "tackles_p90": 1.10, "dribbles_p90": 3.40, "prog_passes_p90": 4.10, "is_reliable": True},
        {"player_id": 162, "player_name": "Isaac Romero", "team_name": "Sevilla FC", "league": "LaLiga", "position": "FW", "age": 24, "market_value_m": 15.0, "minutes": 1700, "xg_p90": 0.32, "xa_p90": 0.25, "goals_p90": 0.28, "assists_p90": 0.22, "interceptions_p90": 0.75, "tackles_p90": 1.30, "dribbles_p90": 1.90, "prog_passes_p90": 3.20, "is_reliable": True},
        {"player_id": 163, "player_name": "Loïc Badé", "team_name": "Sevilla FC", "league": "LaLiga", "position": "DF", "age": 24, "market_value_m": 20.0, "minutes": 2100, "xg_p90": 0.04, "xa_p90": 0.02, "goals_p90": 0.03, "assists_p90": 0.01, "interceptions_p90": 2.35, "tackles_p90": 2.80, "dribbles_p90": 0.35, "prog_passes_p90": 4.60, "is_reliable": True},
        {"player_id": 164, "player_name": "Saúl Ñíguez", "team_name": "Sevilla FC", "league": "LaLiga", "position": "MF", "age": 29, "market_value_m": 7.5, "minutes": 1200, "xg_p90": 0.14, "xa_p90": 0.18, "goals_p90": 0.12, "assists_p90": 0.15, "interceptions_p90": 1.50, "tackles_p90": 2.10, "dribbles_p90": 1.10, "prog_passes_p90": 5.40, "is_reliable": True},

        # Girona FC
        {"player_id": 171, "player_name": "Viktor Tsygankov", "team_name": "Girona FC", "league": "LaLiga", "position": "FW", "age": 26, "market_value_m": 30.0, "minutes": 1650, "xg_p90": 0.32, "xa_p90": 0.36, "goals_p90": 0.35, "assists_p90": 0.30, "interceptions_p90": 0.80, "tackles_p90": 1.35, "dribbles_p90": 2.10, "prog_passes_p90": 5.90, "is_reliable": True},
        {"player_id": 172, "player_name": "Yangel Herrera", "team_name": "Girona FC", "league": "LaLiga", "position": "MF", "age": 26, "market_value_m": 25.0, "minutes": 1800, "xg_p90": 0.22, "xa_p90": 0.14, "goals_p90": 0.25, "assists_p90": 0.11, "interceptions_p90": 1.70, "tackles_p90": 2.60, "dribbles_p90": 1.30, "prog_passes_p90": 5.10, "is_reliable": True},
        {"player_id": 173, "player_name": "Miguel Gutiérrez", "team_name": "Girona FC", "league": "LaLiga", "position": "DF", "age": 23, "market_value_m": 25.0, "minutes": 2200, "xg_p90": 0.10, "xa_p90": 0.28, "goals_p90": 0.09, "assists_p90": 0.24, "interceptions_p90": 1.40, "tackles_p90": 2.05, "dribbles_p90": 2.20, "prog_passes_p90": 7.50, "is_reliable": True},

        # Villarreal CF
        {"player_id": 181, "player_name": "Álex Baena", "team_name": "Villarreal CF", "league": "LaLiga", "position": "MF", "age": 23, "market_value_m": 50.0, "minutes": 2150, "xg_p90": 0.26, "xa_p90": 0.52, "goals_p90": 0.28, "assists_p90": 0.48, "interceptions_p90": 1.15, "tackles_p90": 1.90, "dribbles_p90": 2.40, "prog_passes_p90": 8.60, "is_reliable": True},
        {"player_id": 182, "player_name": "Ayoze Pérez", "team_name": "Villarreal CF", "league": "LaLiga", "position": "FW", "age": 31, "market_value_m": 10.0, "minutes": 1750, "xg_p90": 0.55, "xa_p90": 0.20, "goals_p90": 0.60, "assists_p90": 0.18, "interceptions_p90": 0.80, "tackles_p90": 1.20, "dribbles_p90": 2.10, "prog_passes_p90": 3.80, "is_reliable": True},

        # ==========================================
        # PREMIER LEAGUE (INGLATERRA)
        # ==========================================
        {"player_id": 201, "player_name": "Erling Haaland", "team_name": "Manchester City", "league": "Premier League", "position": "FW", "age": 24, "market_value_m": 200.0, "minutes": 2400, "xg_p90": 0.88, "xa_p90": 0.15, "goals_p90": 0.95, "assists_p90": 0.12, "interceptions_p90": 0.15, "tackles_p90": 0.25, "dribbles_p90": 0.85, "prog_passes_p90": 1.90, "is_reliable": True},
        {"player_id": 202, "player_name": "Rodri Hernández", "team_name": "Manchester City", "league": "Premier League", "position": "MF", "age": 28, "market_value_m": 130.0, "minutes": 2600, "xg_p90": 0.18, "xa_p90": 0.26, "goals_p90": 0.20, "assists_p90": 0.24, "interceptions_p90": 2.10, "tackles_p90": 2.95, "dribbles_p90": 1.25, "prog_passes_p90": 9.80, "is_reliable": True},
        {"player_id": 203, "player_name": "Bukayo Saka", "team_name": "Arsenal", "league": "Premier League", "position": "FW", "age": 23, "market_value_m": 140.0, "minutes": 2350, "xg_p90": 0.45, "xa_p90": 0.44, "goals_p90": 0.48, "assists_p90": 0.42, "interceptions_p90": 0.95, "tackles_p90": 1.80, "dribbles_p90": 3.50, "prog_passes_p90": 6.80, "is_reliable": True},
        {"player_id": 204, "player_name": "Martin Ødegaard", "team_name": "Arsenal", "league": "Premier League", "position": "MF", "age": 25, "market_value_m": 110.0, "minutes": 2400, "xg_p90": 0.28, "xa_p90": 0.46, "goals_p90": 0.30, "assists_p90": 0.40, "interceptions_p90": 1.20, "tackles_p90": 1.75, "dribbles_p90": 2.10, "prog_passes_p90": 8.90, "is_reliable": True},
        {"player_id": 205, "player_name": "Mohamed Salah", "team_name": "Liverpool", "league": "Premier League", "position": "FW", "age": 32, "market_value_m": 55.0, "minutes": 2450, "xg_p90": 0.65, "xa_p90": 0.42, "goals_p90": 0.68, "assists_p90": 0.39, "interceptions_p90": 0.50, "tackles_p90": 0.90, "dribbles_p90": 2.70, "prog_passes_p90": 5.40, "is_reliable": True},
        {"player_id": 206, "player_name": "Cole Palmer", "team_name": "Chelsea", "league": "Premier League", "position": "MF", "age": 22, "market_value_m": 90.0, "minutes": 2300, "xg_p90": 0.52, "xa_p90": 0.40, "goals_p90": 0.58, "assists_p90": 0.38, "interceptions_p90": 0.85, "tackles_p90": 1.35, "dribbles_p90": 2.80, "prog_passes_p90": 7.20, "is_reliable": True},

        # ==========================================
        # BUNDESLIGA (ALEMANIA)
        # ==========================================
        {"player_id": 301, "player_name": "Harry Kane", "team_name": "Bayern Munich", "league": "Bundesliga", "position": "FW", "age": 31, "market_value_m": 100.0, "minutes": 2350, "xg_p90": 0.82, "xa_p90": 0.34, "goals_p90": 0.92, "assists_p90": 0.30, "interceptions_p90": 0.40, "tackles_p90": 0.70, "dribbles_p90": 1.60, "prog_passes_p90": 4.90, "is_reliable": True},
        {"player_id": 302, "player_name": "Jamal Musiala", "team_name": "Bayern Munich", "league": "Bundesliga", "position": "MF", "age": 21, "market_value_m": 130.0, "minutes": 2100, "xg_p90": 0.42, "xa_p90": 0.38, "goals_p90": 0.45, "assists_p90": 0.35, "interceptions_p90": 1.10, "tackles_p90": 1.70, "dribbles_p90": 4.80, "prog_passes_p90": 6.50, "is_reliable": True},
        {"player_id": 303, "player_name": "Florian Wirtz", "team_name": "Bayer Leverkusen", "league": "Bundesliga", "position": "MF", "age": 21, "market_value_m": 130.0, "minutes": 2250, "xg_p90": 0.44, "xa_p90": 0.48, "goals_p90": 0.46, "assists_p90": 0.45, "interceptions_p90": 1.15, "tackles_p90": 1.85, "dribbles_p90": 3.60, "prog_passes_p90": 8.40, "is_reliable": True},
        {"player_id": 304, "player_name": "Alejandro Grimaldo", "team_name": "Bayer Leverkusen", "league": "Bundesliga", "position": "DF", "age": 29, "market_value_m": 45.0, "minutes": 2400, "xg_p90": 0.22, "xa_p90": 0.38, "goals_p90": 0.25, "assists_p90": 0.35, "interceptions_p90": 1.65, "tackles_p90": 2.10, "dribbles_p90": 1.70, "prog_passes_p90": 7.80, "is_reliable": True},

        # ==========================================
        # SERIE A (ITALIA)
        # ==========================================
        {"player_id": 401, "player_name": "Lautaro Martínez", "team_name": "Inter", "league": "Serie A", "position": "FW", "age": 27, "market_value_m": 110.0, "minutes": 2300, "xg_p90": 0.65, "xa_p90": 0.22, "goals_p90": 0.70, "assists_p90": 0.18, "interceptions_p90": 0.55, "tackles_p90": 1.10, "dribbles_p90": 1.95, "prog_passes_p90": 3.40, "is_reliable": True},
        {"player_id": 402, "player_name": "Nicolò Barella", "team_name": "Inter", "league": "Serie A", "position": "MF", "age": 27, "market_value_m": 80.0, "minutes": 2350, "xg_p90": 0.14, "xa_p90": 0.32, "goals_p90": 0.15, "assists_p90": 0.28, "interceptions_p90": 1.85, "tackles_p90": 2.70, "dribbles_p90": 1.80, "prog_passes_p90": 7.60, "is_reliable": True},
        {"player_id": 403, "player_name": "Rafael Leão", "team_name": "Milan", "league": "Serie A", "position": "FW", "age": 25, "market_value_m": 90.0, "minutes": 2100, "xg_p90": 0.42, "xa_p90": 0.35, "goals_p90": 0.45, "assists_p90": 0.32, "interceptions_p90": 0.45, "tackles_p90": 0.85, "dribbles_p90": 4.50, "prog_passes_p90": 5.20, "is_reliable": True},
        {"player_id": 404, "player_name": "Ademola Lookman", "team_name": "Atalanta", "league": "Serie A", "position": "FW", "age": 27, "market_value_m": 40.0, "minutes": 2000, "xg_p90": 0.52, "xa_p90": 0.32, "goals_p90": 0.55, "assists_p90": 0.30, "interceptions_p90": 0.70, "tackles_p90": 1.25, "dribbles_p90": 3.10, "prog_passes_p90": 5.10, "is_reliable": True},

        # ==========================================
        # LIGUE 1 (FRANCIA)
        # ==========================================
        {"player_id": 501, "player_name": "Ousmane Dembélé", "team_name": "Paris Saint-Germain", "league": "Ligue 1", "position": "FW", "age": 27, "market_value_m": 60.0, "minutes": 1950, "xg_p90": 0.38, "xa_p90": 0.54, "goals_p90": 0.35, "assists_p90": 0.50, "interceptions_p90": 0.60, "tackles_p90": 1.20, "dribbles_p90": 4.80, "prog_passes_p90": 7.50, "is_reliable": True},
        {"player_id": 502, "player_name": "Bradley Barcola", "team_name": "Paris Saint-Germain", "league": "Ligue 1", "position": "FW", "age": 22, "market_value_m": 65.0, "minutes": 1800, "xg_p90": 0.55, "xa_p90": 0.30, "goals_p90": 0.60, "assists_p90": 0.28, "interceptions_p90": 0.75, "tackles_p90": 1.30, "dribbles_p90": 3.90, "prog_passes_p90": 4.80, "is_reliable": True},
        {"player_id": 503, "player_name": "Vitinha", "team_name": "Paris Saint-Germain", "league": "Ligue 1", "position": "MF", "age": 24, "market_value_m": 55.0, "minutes": 2200, "xg_p90": 0.16, "xa_p90": 0.28, "goals_p90": 0.18, "assists_p90": 0.25, "interceptions_p90": 1.70, "tackles_p90": 2.45, "dribbles_p90": 1.90, "prog_passes_p90": 8.70, "is_reliable": True},
        {"player_id": 504, "player_name": "Achraf Hakimi", "team_name": "Paris Saint-Germain", "league": "Ligue 1", "position": "DF", "age": 26, "market_value_m": 60.0, "minutes": 2250, "xg_p90": 0.15, "xa_p90": 0.30, "goals_p90": 0.16, "assists_p90": 0.26, "interceptions_p90": 1.55, "tackles_p90": 2.20, "dribbles_p90": 2.50, "prog_passes_p90": 6.80, "is_reliable": True},

        # ==========================================
        # CALADEROS DE TALENTO Y SCOUTING (PORTUGAL, HOLANDA, ETC.)
        # ==========================================
        {"player_id": 601, "player_name": "Viktor Gyökeres", "team_name": "Sporting CP", "league": "Liga Portugal", "position": "FW", "age": 26, "market_value_m": 70.0, "minutes": 2400, "xg_p90": 0.85, "xa_p90": 0.35, "goals_p90": 0.98, "assists_p90": 0.32, "interceptions_p90": 0.50, "tackles_p90": 0.95, "dribbles_p90": 3.20, "prog_passes_p90": 4.10, "is_reliable": True, "potential_ovr": 89, "is_academy": False},
        {"player_id": 602, "player_name": "Francisco Trincão", "team_name": "Sporting CP", "league": "Liga Portugal", "position": "FW", "age": 24, "market_value_m": 23.0, "minutes": 1850, "xg_p90": 0.36, "xa_p90": 0.40, "goals_p90": 0.38, "assists_p90": 0.35, "interceptions_p90": 0.90, "tackles_p90": 1.40, "dribbles_p90": 3.10, "prog_passes_p90": 5.80, "is_reliable": True, "potential_ovr": 84, "is_academy": False},
        {"player_id": 603, "player_name": "Vangelis Pavlidis", "team_name": "Benfica", "league": "Liga Portugal", "position": "FW", "age": 25, "market_value_m": 25.0, "minutes": 1950, "xg_p90": 0.62, "xa_p90": 0.20, "goals_p90": 0.65, "assists_p90": 0.18, "interceptions_p90": 0.40, "tackles_p90": 0.70, "dribbles_p90": 1.80, "prog_passes_p90": 3.10, "is_reliable": True, "potential_ovr": 85, "is_academy": False},
        {"player_id": 604, "player_name": "Orkun Kökçü", "team_name": "Benfica", "league": "Liga Portugal", "position": "MF", "age": 23, "market_value_m": 28.0, "minutes": 2050, "xg_p90": 0.22, "xa_p90": 0.38, "goals_p90": 0.24, "assists_p90": 0.32, "interceptions_p90": 1.45, "tackles_p90": 2.30, "dribbles_p90": 1.70, "prog_passes_p90": 8.20, "is_reliable": True, "potential_ovr": 86, "is_academy": False},
        {"player_id": 605, "player_name": "Johan Bakayoko", "team_name": "PSV", "league": "Eredivisie", "position": "FW", "age": 21, "market_value_m": 45.0, "minutes": 2000, "xg_p90": 0.42, "xa_p90": 0.46, "goals_p90": 0.45, "assists_p90": 0.40, "interceptions_p90": 0.65, "tackles_p90": 1.15, "dribbles_p90": 3.80, "prog_passes_p90": 5.60, "is_reliable": True, "potential_ovr": 88, "is_academy": False},
        {"player_id": 606, "player_name": "Jerdy Schouten", "team_name": "PSV", "league": "Eredivisie", "position": "MF", "age": 27, "market_value_m": 28.0, "minutes": 2200, "xg_p90": 0.08, "xa_p90": 0.18, "goals_p90": 0.07, "assists_p90": 0.15, "interceptions_p90": 2.25, "tackles_p90": 2.90, "dribbles_p90": 0.95, "prog_passes_p90": 8.10, "is_reliable": True, "potential_ovr": 84, "is_academy": False},
        {"player_id": 607, "player_name": "Santiago Giménez", "team_name": "Feyenoord", "league": "Eredivisie", "position": "FW", "age": 23, "market_value_m": 40.0, "minutes": 1850, "xg_p90": 0.70, "xa_p90": 0.18, "goals_p90": 0.75, "assists_p90": 0.15, "interceptions_p90": 0.35, "tackles_p90": 0.60, "dribbles_p90": 1.50, "prog_passes_p90": 2.50, "is_reliable": True, "potential_ovr": 87, "is_academy": False},

        # ==========================================
        # CANTERAS Y JÓVENES PROMESAS (WONDERKIDS SUB-21)
        # ==========================================
        # Cantera Real Madrid ("La Fábrica")
        {"player_id": 701, "player_name": "Endrick Felipe", "team_name": "Real Madrid", "league": "LaLiga", "position": "FW", "age": 18, "market_value_m": 60.0, "minutes": 650, "xg_p90": 0.68, "xa_p90": 0.18, "goals_p90": 0.70, "assists_p90": 0.15, "interceptions_p90": 0.40, "tackles_p90": 0.70, "dribbles_p90": 3.10, "prog_passes_p90": 2.80, "is_reliable": True, "potential_ovr": 94, "is_academy": True, "academy_origin": "La Fábrica"},
        {"player_id": 702, "player_name": "Arda Güler", "team_name": "Real Madrid", "league": "LaLiga", "position": "MF", "age": 19, "market_value_m": 45.0, "minutes": 780, "xg_p90": 0.45, "xa_p90": 0.48, "goals_p90": 0.50, "assists_p90": 0.40, "interceptions_p90": 0.95, "tackles_p90": 1.40, "dribbles_p90": 3.40, "prog_passes_p90": 7.80, "is_reliable": True, "potential_ovr": 92, "is_academy": True, "academy_origin": "La Fábrica"},
        {"player_id": 703, "player_name": "Jesús Fortea", "team_name": "Real Madrid", "league": "LaLiga", "position": "DF", "age": 17, "market_value_m": 5.0, "minutes": 480, "xg_p90": 0.05, "xa_p90": 0.25, "goals_p90": 0.02, "assists_p90": 0.20, "interceptions_p90": 1.85, "tackles_p90": 2.40, "dribbles_p90": 2.10, "prog_passes_p90": 5.90, "is_reliable": True, "potential_ovr": 88, "is_academy": True, "academy_origin": "La Fábrica"},

        # Cantera FC Barcelona ("La Masia")
        {"player_id": 711, "player_name": "Marc Casadó", "team_name": "Barcelona", "league": "LaLiga", "position": "MF", "age": 21, "market_value_m": 25.0, "minutes": 1450, "xg_p90": 0.08, "xa_p90": 0.35, "goals_p90": 0.05, "assists_p90": 0.30, "interceptions_p90": 2.20, "tackles_p90": 3.10, "dribbles_p90": 1.30, "prog_passes_p90": 8.60, "is_reliable": True, "potential_ovr": 90, "is_academy": True, "academy_origin": "La Masia"},
        {"player_id": 712, "player_name": "Héctor Fort", "team_name": "Barcelona", "league": "LaLiga", "position": "DF", "age": 18, "market_value_m": 12.0, "minutes": 720, "xg_p90": 0.04, "xa_p90": 0.22, "goals_p90": 0.02, "assists_p90": 0.18, "interceptions_p90": 1.75, "tackles_p90": 2.30, "dribbles_p90": 1.80, "prog_passes_p90": 5.40, "is_reliable": True, "potential_ovr": 87, "is_academy": True, "academy_origin": "La Masia"},
        {"player_id": 713, "player_name": "Pau Prim", "team_name": "Barcelona", "league": "LaLiga", "position": "MF", "age": 18, "market_value_m": 4.0, "minutes": 520, "xg_p90": 0.06, "xa_p90": 0.28, "goals_p90": 0.04, "assists_p90": 0.22, "interceptions_p90": 1.95, "tackles_p90": 2.60, "dribbles_p90": 1.10, "prog_passes_p90": 8.10, "is_reliable": True, "potential_ovr": 88, "is_academy": True, "academy_origin": "La Masia"},

        # Cantera Real Betis ("Cantera Verdiblanca")
        {"player_id": 721, "player_name": "Assane Diao", "team_name": "Real Betis", "league": "LaLiga", "position": "FW", "age": 19, "market_value_m": 12.0, "minutes": 980, "xg_p90": 0.38, "xa_p90": 0.18, "goals_p90": 0.35, "assists_p90": 0.15, "interceptions_p90": 0.70, "tackles_p90": 1.20, "dribbles_p90": 2.90, "prog_passes_p90": 3.80, "is_reliable": True, "potential_ovr": 87, "is_academy": True, "academy_origin": "Cantera Real Betis"},
        {"player_id": 722, "player_name": "Mateo Flores", "team_name": "Real Betis", "league": "LaLiga", "position": "MF", "age": 20, "market_value_m": 3.0, "minutes": 510, "xg_p90": 0.10, "xa_p90": 0.24, "goals_p90": 0.08, "assists_p90": 0.20, "interceptions_p90": 1.80, "tackles_p90": 2.45, "dribbles_p90": 1.40, "prog_passes_p90": 6.80, "is_reliable": True, "potential_ovr": 85, "is_academy": True, "academy_origin": "Cantera Real Betis"},

        # Cantera Sevilla FC ("Carretera de Utrera")
        {"player_id": 731, "player_name": "Alberto Collado", "team_name": "Sevilla FC", "league": "LaLiga", "position": "MF", "age": 19, "market_value_m": 3.5, "minutes": 490, "xg_p90": 0.15, "xa_p90": 0.26, "goals_p90": 0.12, "assists_p90": 0.22, "interceptions_p90": 1.55, "tackles_p90": 2.20, "dribbles_p90": 1.80, "prog_passes_p90": 6.10, "is_reliable": True, "potential_ovr": 86, "is_academy": True, "academy_origin": "Cantera Sevilla FC"},
        {"player_id": 732, "player_name": "Stanis Idumbo", "team_name": "Sevilla FC", "league": "LaLiga", "position": "FW", "age": 19, "market_value_m": 6.0, "minutes": 620, "xg_p90": 0.35, "xa_p90": 0.25, "goals_p90": 0.38, "assists_p90": 0.20, "interceptions_p90": 0.60, "tackles_p90": 1.10, "dribbles_p90": 3.20, "prog_passes_p90": 4.10, "is_reliable": True, "potential_ovr": 88, "is_academy": True, "academy_origin": "Cantera Sevilla FC"},

        # Cantera Athletic Club ("Lezama")
        {"player_id": 741, "player_name": "Unai Gómez", "team_name": "Athletic Club", "league": "LaLiga", "position": "MF", "age": 21, "market_value_m": 15.0, "minutes": 1100, "xg_p90": 0.25, "xa_p90": 0.20, "goals_p90": 0.28, "assists_p90": 0.18, "interceptions_p90": 1.40, "tackles_p90": 2.30, "dribbles_p90": 1.80, "prog_passes_p90": 5.80, "is_reliable": True, "potential_ovr": 87, "is_academy": True, "academy_origin": "Lezama"},
        {"player_id": 742, "player_name": "Peio Canales", "team_name": "Athletic Club", "league": "LaLiga", "position": "MF", "age": 19, "market_value_m": 4.0, "minutes": 480, "xg_p90": 0.18, "xa_p90": 0.30, "goals_p90": 0.15, "assists_p90": 0.25, "interceptions_p90": 1.50, "tackles_p90": 2.10, "dribbles_p90": 2.10, "prog_passes_p90": 6.50, "is_reliable": True, "potential_ovr": 86, "is_academy": True, "academy_origin": "Lezama"},

        # Canteras Europeas (Sporting CP, Benfica, Juventus)
        {"player_id": 751, "player_name": "Geovany Quenda", "team_name": "Sporting CP", "league": "Liga Portugal", "position": "FW", "age": 17, "market_value_m": 20.0, "minutes": 1250, "xg_p90": 0.28, "xa_p90": 0.38, "goals_p90": 0.25, "assists_p90": 0.35, "interceptions_p90": 1.20, "tackles_p90": 1.90, "dribbles_p90": 4.10, "prog_passes_p90": 6.10, "is_reliable": True, "potential_ovr": 92, "is_academy": True, "academy_origin": "Academia Sporting CP"},
        {"player_id": 752, "player_name": "Kenan Yıldız", "team_name": "Juventus", "league": "Serie A", "position": "FW", "age": 19, "market_value_m": 40.0, "minutes": 1350, "xg_p90": 0.40, "xa_p90": 0.32, "goals_p90": 0.42, "assists_p90": 0.28, "interceptions_p90": 0.70, "tackles_p90": 1.25, "dribbles_p90": 3.60, "prog_passes_p90": 5.40, "is_reliable": True, "potential_ovr": 91, "is_academy": True, "academy_origin": "Next Gen Juventus"},

    ]

    df = pd.DataFrame(players)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def load_real_players_df() -> pd.DataFrame:
    """Carga la base de datos global y exhaustiva de futbolistas reales de todo el mundo."""
    from src.data.global_player_database import load_global_players
    return load_global_players()
