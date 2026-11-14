import os
import pytest
import pandas as pd
from src.scrapers.football_data import FootballDataScraper
from src.scrapers.fbref import FBrefScraper
from src.processing.db_loader import DBLoader
from src.config.db_init import init_db, get_connection

@pytest.fixture
def sample_csv_path(tmp_path):
    """Crea un archivo CSV temporal simulando la respuesta de football-data.co.uk."""
    csv_content = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
        "15/08/2025,Real Madrid,Barcelona,2,1,2.10,3.40,3.20\n"
        "16/08/2025,Atletico Madrid,Sevilla,1,1,1.80,3.50,4.50\n"
    )
    file_path = tmp_path / "test_laliga.csv"
    file_path.write_text(csv_content, encoding="utf-8")
    return str(file_path)

@pytest.fixture
def sample_fbref_html():
    """Genera un fragmento de HTML simulando la tabla 'stats_standard' de FBref."""
    return """
    <html>
        <body>
            <table id="stats_standard">
                <thead>
                    <tr class="thead"><th>Player</th><th>Team</th><th>Pos</th><th>Age</th><th>Min</th><th>Gls</th><th>Ast</th><th>xG</th><th>xAG</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td data-stat="player">Kylian Mbappé</td>
                        <td data-stat="team">Real Madrid</td>
                        <td data-stat="position">FW</td>
                        <td data-stat="age">27</td>
                        <td data-stat="minutes">1,800</td>
                        <td data-stat="goals">18</td>
                        <td data-stat="assists">5</td>
                        <td data-stat="xg">16.4</td>
                        <td data-stat="xg_assist">4.2</td>
                    </tr>
                    <tr>
                        <td data-stat="player">Lamine Yamal</td>
                        <td data-stat="team">Barcelona</td>
                        <td data-stat="position">FW</td>
                        <td data-stat="age">18</td>
                        <td data-stat="minutes">1,650</td>
                        <td data-stat="goals">8</td>
                        <td data-stat="assists">10</td>
                        <td data-stat="xg">7.1</td>
                        <td data-stat="xg_assist">9.5</td>
                    </tr>
                </tbody>
            </table>
        </body>
    </html>
    """

def test_parse_football_data_csv(sample_csv_path):
    """Verifica que el parser de football-data transforma correctamente las columnas y tipos."""
    scraper = FootballDataScraper(raw_data_dir="/tmp")
    df = scraper.parse_csv(sample_csv_path)

    assert len(df) == 2, "El DataFrame debería contener exactamente 2 partidos."
    assert "home_team" in df.columns
    assert df.iloc[0]["home_team"] == "Real Madrid"
    assert df.iloc[0]["home_goals"] == 2
    assert df.iloc[0]["odds_home"] == 2.10

def test_parse_fbref_html(sample_fbref_html):
    """Verifica el parseo de HTML de FBref extrayendo métricas clave de jugadores."""
    scraper = FBrefScraper(raw_data_dir="/tmp")
    df = scraper.parse_player_stats_html(sample_fbref_html)

    assert len(df) == 2, "Debería parsear exactamente 2 jugadores."
    assert df.iloc[0]["player_name"] == "Kylian Mbappé"
    assert df.iloc[0]["minutes"] == 1800
    assert df.iloc[0]["goals"] == 18
    assert df.iloc[0]["xg"] == 16.4

def test_db_loader_ingestion(sample_csv_path, sample_fbref_html):
    """Prueba de integración end-to-end: parseo e ingesta en PostgreSQL."""
    init_db()  # Re-asegurar esquema
    loader = DBLoader()
    
    # 1. Ingesta de partidos
    scraper_fd = FootballDataScraper(raw_data_dir="/tmp")
    df_matches = scraper_fd.parse_csv(sample_csv_path)
    inserted_matches = loader.load_matches_df(df_matches, season="2526")
    assert inserted_matches == 2

    # 2. Ingesta de jugadores
    scraper_fb = FBrefScraper(raw_data_dir="/tmp")
    df_players = scraper_fb.parse_player_stats_html(sample_fbref_html)
    inserted_players = loader.load_player_stats_df(df_players)
    assert inserted_players == 2

    # 3. Comprobar persisted data en PostgreSQL
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fact_matches;")
    matches_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM dim_players;")
    players_count = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    assert matches_count >= 2
    assert players_count >= 2