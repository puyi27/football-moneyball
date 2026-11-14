import os
import requests
import pandas as pd

class FootballDataScraper:
    """Extractor de datos de calendario, resultados y cuotas desde football-data.co.uk."""
    
    BASE_URL = "https://www.football-data.co.uk/mmz4281"
    # SP1 es el código de LaLiga Primera División en football-data.co.uk
    LIGA_CODES = {"LaLiga": "SP1"}

    def __init__(self, raw_data_dir="data/raw"):
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def download_csv(self, season_code="2526", league="LaLiga"):
        """
        Descarga el CSV de la temporada indicada.
        season_code ejemplo: '2526' para 2025/2026, '2627' para 2026/2027.
        """
        code = self.LIGA_CODES.get(league, "SP1")
        url = f"{self.BASE_URL}/{season_code}/{code}.csv"
        file_path = os.path.join(self.raw_data_dir, f"football_data_{league}_{season_code}.csv")

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"CSV descargado exitosamente: {file_path}")
                return file_path
            else:
                print(f"Error al descargar CSV de {url}. Status Code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Excepción al descargar {url}: {e}")
            return None

    def parse_csv(self, file_path):
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
            'B365A': 'odds_away'
        }
        
        # Filtrar solo columnas presentes
        df_filtered = df[[col for col in cols_required.keys() if col in df.columns]].copy()
        df_filtered.rename(columns=cols_required, inplace=True)
        
        # Formatear fechas
        if 'match_date' in df_filtered.columns:
            df_filtered['match_date'] = pd.to_datetime(df_filtered['match_date'], dayfirst=True, errors='coerce')
        
        return df_filtered.dropna(subset=['home_team', 'away_team', 'home_goals', 'away_goals'])