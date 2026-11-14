import os
import requests
import pandas as pd
from typing import Dict, List, Optional

class FootballDataScraper:
    """
    Extractor de datos de calendario, resultados y cuotas desde football-data.co.uk
    con cobertura para las 5 Grandes Ligas europeas y Ligas de Scouting secundarias.
    """
    
    BASE_URL = "https://www.football-data.co.uk/mmz4281"

    # Mapeo oficial de códigos de división de football-data.co.uk
    LIGA_CODES = {
        # Top 5 Grandes Ligas Europeas
        "LaLiga": "SP1",
        "Premier League": "E0",
        "Serie A": "I1",
        "Bundesliga": "D1",
        "Ligue 1": "F1",

        # Segundas divisiones de las Top 5 (gran caladero de scouting)
        "LaLiga 2": "SP2",
        "Championship": "E1",
        "Serie B": "I2",
        "2. Bundesliga": "D2",
        "Ligue 2": "F2",

        # Ligas de desarrollo y talento ("Feeder / Scouting Leagues")
        "Liga Portugal": "P1",
        "Eredivisie": "N1",
        "Belgian Pro League": "B1",
        "Scottish Premiership": "SC0",
        "Super Lig": "T1",
        "Greek Super League": "G1",
    }

    TOP_5_LEAGUES = ["LaLiga", "Premier League", "Serie A", "Bundesliga", "Ligue 1"]
    SCOUTING_LEAGUES = ["Liga Portugal", "Eredivisie", "Belgian Pro League", "Championship", "LaLiga 2"]

    def __init__(self, raw_data_dir="data/raw"):
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def get_supported_leagues(self) -> List[str]:
        """Devuelve la lista de competiciones soportadas."""
        return list(self.LIGA_CODES.keys())

    def download_csv(self, season_code="2526", league="LaLiga") -> Optional[str]:
        """
        Descarga el CSV de la temporada y liga indicadas.
        season_code ejemplo: '2425' para 2024/2025, '2526' para 2025/2026.
        """
        code = self.LIGA_CODES.get(league)
        if not code:
            raise ValueError(f"Liga '{league}' no soportada. Ligas disponibles: {list(self.LIGA_CODES.keys())}")

        url = f"{self.BASE_URL}/{season_code}/{code}.csv"
        safe_league_name = league.replace(" ", "_").lower()
        file_path = os.path.join(self.raw_data_dir, f"football_data_{safe_league_name}_{season_code}.csv")

        try:
            response = requests.get(url, timeout=12)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"[{league}] CSV descargado exitosamente: {file_path}")
                return file_path
            else:
                print(f"Error al descargar {league} desde {url}. Código HTTP: {response.status_code}")
                return None
        except Exception as e:
            print(f"Excepción al descargar {league} ({url}): {e}")
            return None

    def download_leagues_batch(self, leagues: List[str], season_code="2526") -> Dict[str, Optional[str]]:
        """Descarga en lote un conjunto de ligas seleccionadas."""
        results = {}
        for league in leagues:
            file_path = self.download_csv(season_code=season_code, league=league)
            results[league] = file_path
        return results

    def download_top5_leagues(self, season_code="2526") -> Dict[str, Optional[str]]:
        """Descarga los datos de las 5 Grandes Ligas Europeas."""
        return self.download_leagues_batch(self.TOP_5_LEAGUES, season_code=season_code)

    def download_scouting_leagues(self, season_code="2526") -> Dict[str, Optional[str]]:
        """Descarga los datos de las principales ligas de desarrollo y scouting."""
        return self.download_leagues_batch(self.SCOUTING_LEAGUES, season_code=season_code)

    def parse_csv(self, file_path: str, league: str = "LaLiga") -> pd.DataFrame:
        """Parsea y limpia los campos principales del CSV de football-data."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe.")

        df = pd.read_csv(file_path, encoding="unicode_escape")
        
        # Mapeo de columnas estándar
        cols_required = {
            'Date': 'match_date',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'home_goals',
            'FTAG': 'away_goals',
            'B365H': 'odds_home',
            'B365D': 'odds_draw',
            'B365A': 'odds_away',
            'HST': 'home_shots_target',
            'AST': 'away_shots_target',
            'HC': 'home_corners',
            'AC': 'away_corners'
        }
        
        # Filtrar solo columnas presentes
        present_cols = [col for col in cols_required.keys() if col in df.columns]
        df_filtered = df[present_cols].copy()
        df_filtered.rename(columns={c: cols_required[c] for c in present_cols}, inplace=True)
        
        # Inyectar etiqueta de liga
        df_filtered['league'] = league

        # Formatear fechas
        if 'match_date' in df_filtered.columns:
            df_filtered['match_date'] = pd.to_datetime(df_filtered['match_date'], dayfirst=True, errors='coerce')
        
        return df_filtered.dropna(subset=['home_team', 'away_team', 'home_goals', 'away_goals'])
