"""
Scrapers Package (Deprecated).

Este módulo provee retrocompatibilidad con los extractores heredados de FBref y Football-Data.
Para nuevos desarrollos, utilice `src.data.advanced_sources_hub.MasterAnalyticsDataHub`.
"""

from src.scrapers.fbref import FBrefScraper
from src.scrapers.football_data import FootballDataScraper

__all__ = ["FBrefScraper", "FootballDataScraper"]
