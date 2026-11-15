"""
Football Data Scraper (Deprecated).

Este módulo ha sido deprecado en favor de la arquitectura unificada en
`src.data.advanced_sources_hub.MasterAnalyticsDataHub` y `OpenFootballFeed`.
Se mantiene esta clase exclusivamente para retrocompatibilidad con los tests y
cargas existentes.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import ClassVar

import pandas as pd
import requests

from src.data.advanced_sources_hub import MasterAnalyticsDataHub, OpenFootballFeed

logger = logging.getLogger(__name__)


class FootballDataScraper:
    """
    DEPRECADO: Utilice MasterAnalyticsDataHub o OpenFootballFeed en src.data.advanced_sources_hub.
    Extractor de calendarios, resultados y cuotas con retrocompatibilidad.
    """

    BASE_URL: ClassVar[str] = "https://www.football-data.co.uk/mmz4281"

    # Mapeo oficial de códigos de división de football-data.co.uk
    LIGA_CODES: ClassVar[dict[str, str]] = {
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

    TOP_5_LEAGUES: ClassVar[list[str]] = [
        "LaLiga",
        "Premier League",
        "Serie A",
        "Bundesliga",
        "Ligue 1",
    ]
    SCOUTING_LEAGUES: ClassVar[list[str]] = [
        "Liga Portugal",
        "Eredivisie",
        "Belgian Pro League",
        "Championship",
        "LaLiga 2",
    ]

    def __init__(self, raw_data_dir: str = "data/raw") -> None:
        warnings.warn(
            "FootballDataScraper está deprecado y será eliminado en futuras versiones. "
            "Utilice src.data.advanced_sources_hub.OpenFootballFeed o MasterAnalyticsDataHub.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)
        # Delegación en el feed unificado
        self._hub = OpenFootballFeed()

    def get_unified_hub(self) -> MasterAnalyticsDataHub:
        """Acceso directo al hub de analítica deportiva unificada."""
        return MasterAnalyticsDataHub()

    def get_supported_leagues(self) -> list[str]:
        """Devuelve la lista de competiciones soportadas."""
        return list(self.LIGA_CODES.keys())

    def download_csv(
        self, season_code: str = "2526", league: str = "LaLiga"
    ) -> str | None:
        """
        Descarga el CSV de la temporada y liga indicadas.
        season_code ejemplo: '2425' para 2024/2025, '2526' para 2025/2026.
        """
        code = self.LIGA_CODES.get(league)
        if not code:
            raise ValueError(
                f"Liga '{league}' no soportada. Ligas disponibles: {list(self.LIGA_CODES.keys())}"
            )

        url = f"{self.BASE_URL}/{season_code}/{code}.csv"
        safe_league_name = league.replace(" ", "_").lower()
        file_path = os.path.join(
            self.raw_data_dir, f"football_data_{safe_league_name}_{season_code}.csv"
        )

        try:
            response = requests.get(url, timeout=12)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                logger.info("[%s] CSV descargado exitosamente: %s", league, file_path)
                return file_path
            logger.warning(
                "Error al descargar %s desde %s. Código HTTP: %s",
                league,
                url,
                response.status_code,
            )
            return None
        except requests.RequestException as e:
            logger.error("Excepción al descargar %s (%s): %s", league, url, e)
            return None

    def download_leagues_batch(
        self, leagues: list[str], season_code: str = "2526"
    ) -> dict[str, str | None]:
        """Descarga en lote un conjunto de ligas seleccionadas."""
        results: dict[str, str | None] = {}
        for league in leagues:
            file_path = self.download_csv(season_code=season_code, league=league)
            results[league] = file_path
        return results

    def download_top5_leagues(self, season_code: str = "2526") -> dict[str, str | None]:
        """Descarga los datos de las 5 Grandes Ligas Europeas."""
        return self.download_leagues_batch(self.TOP_5_LEAGUES, season_code=season_code)

    def download_scouting_leagues(
        self, season_code: str = "2526"
    ) -> dict[str, str | None]:
        """Descarga los datos de las principales ligas de desarrollo y scouting."""
        return self.download_leagues_batch(
            self.SCOUTING_LEAGUES, season_code=season_code
        )

    def parse_csv(self, file_path: str, league: str = "LaLiga") -> pd.DataFrame:
        """Parsea y limpia los campos principales del CSV de football-data."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe.")

        df = pd.read_csv(file_path, encoding="unicode_escape")

        # Mapeo de columnas estándar
        cols_required = {
            "Date": "match_date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "B365H": "odds_home",
            "B365D": "odds_draw",
            "B365A": "odds_away",
            "HST": "home_shots_target",
            "AST": "away_shots_target",
            "HC": "home_corners",
            "AC": "away_corners",
        }

        # Filtrar solo columnas presentes
        present_cols = [col for col in cols_required if col in df.columns]
        df_filtered = df[present_cols].copy()
        df_filtered.rename(
            columns={c: cols_required[c] for c in present_cols}, inplace=True
        )

        # Inyectar etiqueta de liga
        df_filtered["league"] = league

        # Formatear fechas
        if "match_date" in df_filtered.columns:
            df_filtered["match_date"] = pd.to_datetime(
                df_filtered["match_date"], dayfirst=True, errors="coerce"
            )

        return df_filtered.dropna(
            subset=["home_team", "away_team", "home_goals", "away_goals"]
        )

    def get_league_season_matches(
        self, league_code: str = "es.1", season: str = "2023-24"
    ) -> pd.DataFrame:
        """Método delegado en OpenFootballFeed para calendarios y resultados normalizados."""
        return self._hub.get_league_season_matches(
            league_code=league_code, season=season
        )
