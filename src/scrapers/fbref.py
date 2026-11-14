import os
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

class FBrefScraper:
    """Extractor de métricas de jugadores desde FBref con caching y pausas respetuosas."""

    def __init__(self, raw_data_dir="data/raw", delay=3):
        self.raw_data_dir = raw_data_dir
        self.delay = delay
        os.makedirs(self.raw_data_dir, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

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

    def parse_player_stats_html(self, html_content):
        """Parsea la tabla de estadísticas estándar de jugadores en el HTML de FBref."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Buscar la tabla de estadísticas estándar de jugadores
        table = soup.find("table", {"id": lambda x: x and "stats_standard" in x})
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

            rows.append({
                "player_name": player_name,
                "team_name": team_name,
                "position": position,
                "age": age,
                "minutes": pd.to_numeric(minutes, errors='coerce') or 0,
                "goals": pd.to_numeric(goals, errors='coerce') or 0,
                "assists": pd.to_numeric(assists, errors='coerce') or 0,
                "xg": pd.to_numeric(xg, errors='coerce') or 0.0,
                "xa": pd.to_numeric(xa, errors='coerce') or 0.0,
            })

        return pd.DataFrame(rows)