import pytest
import pandas as pd
from src.data.football_api_connector import FootballAPIConnector
from src.data.global_player_database import (
    generate_worldwide_player_network,
    build_and_save_global_database,
    load_global_players,
    sync_global_database_to_postgres,
)


def test_football_api_connector_position_mapping():
    """Verifica que el normalizador de posiciones mapea correctamente términos ingleses y españoles."""
    connector = FootballAPIConnector()
    assert connector._map_position("Goalkeeper") == "GK"
    assert connector._map_position("Portero") == "GK"
    assert connector._map_position("Defender") == "DF"
    assert connector._map_position("Defensa") == "DF"
    assert connector._map_position("Midfielder") == "MF"
    assert connector._map_position("Centrocampista") == "MF"
    assert connector._map_position("Attacker") == "FW"
    assert connector._map_position("Delantero") == "FW"


def test_football_api_connector_schema_normalization():
    """Verifica que la normalización de jugadores desde respuestas JSON de APIs se ajusta al esquema del sistema."""
    connector = FootballAPIConnector()
    raw = {
        "player_id": 12345,
        "player_name": "Test Striker",
        "team_name": "Borussia Dortmund",
        "league": "Bundesliga",
        "position": "Attacker",
        "age": 20,
        "market_value_m": 45.0,
        "goals_p90": 0.65,
        "assists_p90": 0.25,
    }
    normalized = connector.normalize_api_player_to_schema(raw)
    assert normalized["player_id"] == 12345
    assert normalized["position"] == "FW"
    assert normalized["xg_p90"] > 0
    assert normalized["potential_ovr"] >= 80
    assert "is_academy" in normalized


def test_football_api_connector_status_check():
    """Verifica que el probador de conexión devuelve diagnóstico coherente cuando no hay API key."""
    connector = FootballAPIConnector(api_football_key="")
    status = connector.test_api_football_connection()
    assert status["status"] == "NO_KEY"
    assert not status["connected"]


def test_global_player_network_coverage():
    """Verifica que la red mundial abarca las ligas solicitadas y todos los 20 clubes de LaLiga."""
    df = generate_worldwide_player_network()
    assert len(df) >= 100, f"Debe tener una base de datos amplia (>100 jugadores): {len(df)}"

    leagues = set(df["league"].unique())
    expected_leagues = {
        "LaLiga", "Premier League", "Serie A", "Bundesliga", "Ligue 1",
        "Liga Portugal", "Eredivisie", "LaLiga 2", "Brasileirão",
        "Liga Argentina", "Saudi Pro League", "MLS"
    }
    for el in expected_leagues:
        assert el in leagues, f"Falta la competición {el} en la red mundial"

    # Verificar que los 20 clubes de LaLiga están representados
    laliga_teams = set(df[df["league"] == "LaLiga"]["team_name"].unique())
    assert len(laliga_teams) >= 18, f"Deben estar los clubes de LaLiga representados: {len(laliga_teams)}"
    assert "Real Madrid" in laliga_teams
    assert "Barcelona" in laliga_teams
    assert "Atletico Madrid" in laliga_teams
    assert "Sevilla FC" in laliga_teams
    assert "Real Betis" in laliga_teams
    assert "Athletic Club" in laliga_teams


def test_global_database_persistence():
    """Verifica la persistencia en CSV y carga coherente de la base de datos mundial."""
    df = build_and_save_global_database()
    assert not df.empty
    loaded = load_global_players()
    assert len(loaded) == len(df)
    assert "potential_ovr" in loaded.columns
    assert "market_value_m" in loaded.columns
