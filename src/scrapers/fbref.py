import os
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Optional

class FBrefScraper:
    """
    Extractor de métricas avanzadas de jugadores desde FBref con caching local,
    pausas respetuosas y soporte para las 5 Grandes Ligas y ligas de scouting.
    """

    # URLs base oficiales de competiciones en FBref
    LEAGUE_URLS = {
        # Compilación de las 5 Grandes Ligas (Big 5)
        "Big5": "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats",
        
        # Ligas individuales principales
        "LaLiga": "https://fbref.com/en/comps/12/stats/La-Liga-Stats",
        "Premier League": "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
        "Serie A": "https://fbref.com/en/comps/11/stats/Serie-A-Stats",
        "Bundesliga": "https://fbref.com/en/comps/20/stats/Bundesliga-Stats",
        "Ligue 1": "https://fbref.com/en/comps/13/stats/Ligue-1-Stats",

        # Ligas de desarrollo / Scouting
        "Liga Portugal": "https://fbref.com/en/comps/32/stats/Primeira-Liga-Stats",
        "Eredivisie": "https://fbref.com/en/comps/23/stats/Eredivisie-Stats",
        "Belgian Pro League": "https://fbref.com/en/comps/37/stats/Belgian-Pro-League-Stats",
        "Championship": "https://fbref.com/en/comps/10/stats/Championship-Stats",
        "LaLiga 2": "https://fbref.com/en/comps/17/stats/Segunda-Division-Stats"
    }

    TOP_5_LEAGUES = ["LaLiga", "Premier League", "Serie A", "Bundesliga", "Ligue 1"]
    SCOUTING_LEAGUES = ["Liga Portugal", "Eredivisie", "Belgian Pro League", "Championship", "LaLiga 2"]

    def __init__(self, raw_data_dir="data/raw", delay=3):
        self.raw_data_dir = raw_data_dir
        self.delay = delay
        os.makedirs(self.raw_data_dir, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def get_supported_leagues(self) -> List[str]:
        """Devuelve las competiciones soportadas para extracción."""
        return list(self.LEAGUE_URLS.keys())

    def fetch_page(self, url, cache_filename):
        """Descarga una página HTML con caché local para evitar peticiones repetidas."""
        cache_path = os.path.join(self.raw_data_dir, cache_filename)
        
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

        time.sleep(self.delay)
        response = requests.get(url, headers=self.headers, timeout=15)
        if response.status_code == 200:
            html_content = response.text
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return html_content
        else:
            raise RuntimeError(f"Error {response.status_code} al solicitar la URL: {url}")

    def fetch_league_stats(self, league: str = "LaLiga", stat_type: str = "stats") -> str:
        """
        Descarga el HTML de estadísticas para una liga determinada.
        stat_type puede ser: 'stats' (estándar), 'defense', 'passing', 'possession'.
        """
        base_url = self.LEAGUE_URLS.get(league)
        if not base_url:
            raise ValueError(f"Liga '{league}' no soportada. Disponibles: {list(self.LEAGUE_URLS.keys())}")

        if stat_type != "stats":
            # Sustituye /stats/ por el tipo solicitado (ej. /defense/, /passing/)
            url = base_url.replace("/stats/", f"/{stat_type}/")
        else:
            url = base_url

        safe_league = league.replace(" ", "_").lower()
        cache_filename = f"fbref_{safe_league}_{stat_type}.html"
        return self.fetch_page(url, cache_filename)

    def parse_player_stats_html(self, html_content: str, default_league: str = "LaLiga") -> pd.DataFrame:
        """
        Parsea la tabla de estadísticas estándar de jugadores en el HTML de FBref,
        extrayendo métricas ofensivas y defensivas fundamentales.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Buscar la tabla de estadísticas estándar de jugadores
        table = soup.find("table", {"id": lambda x: x and ("stats_standard" in x or "stats_defense" in x)})
        if not table:
            return pd.DataFrame()

        rows = []
        for tr in table.find_all("tr"):
            if tr.get("class") and ("thead" in tr["class"] or "header" in tr["class"]):
                continue
            
            player_cell = tr.find("td", {"data-stat": "player"})
            if not player_cell:
                continue

            player_name = player_cell.get_text(strip=True)
            team_cell = tr.find("td", {"data-stat": "team"})
            team_name = team_cell.get_text(strip=True) if team_cell else "Desconocido"
            
            # Liga o competición de origen si viene en tabla combinada (ej. Big 5)
            comp_cell = tr.find("td", {"data-stat": "comp_level"})
            league = comp_cell.get_text(strip=True) if comp_cell else default_league

            pos_cell = tr.find("td", {"data-stat": "position"})
            position = pos_cell.get_text(strip=True) if pos_cell else None
            
            age_cell = tr.find("td", {"data-stat": "age"})
            age = age_cell.get_text(strip=True) if age_cell else None
            
            min_cell = tr.find("td", {"data-stat": "minutes"})
            minutes = min_cell.get_text(strip=True).replace(",", "") if min_cell else "0"
            
            gls_cell = tr.find("td", {"data-stat": "goals"})
            goals = gls_cell.get_text(strip=True) if gls_cell else "0"
            
            ast_cell = tr.find("td", {"data-stat": "assists"})
            assists = ast_cell.get_text(strip=True) if ast_cell else "0"

            xg_cell = tr.find("td", {"data-stat": "xg"})
            xg = xg_cell.get_text(strip=True) if xg_cell else "0.0"

            xa_cell = tr.find("td", {"data-stat": "xg_assist"})
            xa = xa_cell.get_text(strip=True) if xa_cell else "0.0"

            # Métricas defensivas / recuperación si están presentes
            tkl_cell = tr.find("td", {"data-stat": "tackles"})
            tackles = tkl_cell.get_text(strip=True) if tkl_cell else "0"

            int_cell = tr.find("td", {"data-stat": "interceptions"})
            interceptions = int_cell.get_text(strip=True) if int_cell else "0"

            rows.append({
                "player_name": player_name,
                "team_name": team_name,
                "league": league,
                "position": position,
                "age": pd.to_numeric(age, errors='coerce') if age else None,
                "minutes": pd.to_numeric(minutes, errors='coerce') or 0,
                "goals": pd.to_numeric(goals, errors='coerce') or 0,
                "assists": pd.to_numeric(assists, errors='coerce') or 0,
                "xg": pd.to_numeric(xg, errors='coerce') or 0.0,
                "xa": pd.to_numeric(xa, errors='coerce') or 0.0,
                "tackles": pd.to_numeric(tackles, errors='coerce') or 0,
                "interceptions": pd.to_numeric(interceptions, errors='coerce') or 0,
            })

        df = pd.DataFrame(rows)
        return df

    def parse_defense_stats_html(self, html_content: str) -> pd.DataFrame:
        """Parsea la tabla específica de defensa para extraer tackles, intercepciones y bloqueos."""
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table", {"id": lambda x: x and "stats_defense" in x})
        if not table:
            return pd.DataFrame()

        rows = []
        for tr in table.find_all("tr"):
            if tr.get("class") and ("thead" in tr["class"] or "header" in tr["class"]):
                continue

            p_cell = tr.find("td", {"data-stat": "player"})
            t_cell = tr.find("td", {"data-stat": "team"})
            if not p_cell:
                continue

            tackles = tr.find("td", {"data-stat": "tackles"})
            interceptions = tr.find("td", {"data-stat": "interceptions"})
            blocks = tr.find("td", {"data-stat": "blocks"})

            rows.append({
                "player_name": p_cell.get_text(strip=True),
                "team_name": t_cell.get_text(strip=True) if t_cell else "Desconocido",
                "tackles": pd.to_numeric(tackles.get_text(strip=True), errors='coerce') if tackles else 0,
                "interceptions": pd.to_numeric(interceptions.get_text(strip=True), errors='coerce') if interceptions else 0,
                "blocks": pd.to_numeric(blocks.get_text(strip=True), errors='coerce') if blocks else 0,
            })

        return pd.DataFrame(rows)
