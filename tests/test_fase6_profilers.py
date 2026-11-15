import pytest
import pandas as pd
import numpy as np
from src.data.real_dataset_loader import load_real_players_df
from src.models_scouting.player_profiler import PlayerProfiler
from src.models_analytics.team_profiler import TeamProfiler
from src.models_analytics.elo_model import EloSystem


@pytest.fixture
def real_df():
    return load_real_players_df()


def test_fifa_card_attributes(real_df):
    """Verifica que los atributos FIFA (PAC, SHO, PAS, DRI, DEF, PHY y OVR) se calculan entre 45 y 99."""
    profiler = PlayerProfiler(real_df)
    vini = real_df[real_df["player_name"] == "Vinícius Júnior"].iloc[0]
    card = profiler.calculate_fifa_card(vini)

    assert card["OVR"] >= 88, f"OVR de Vinícius debería ser élite: {card['OVR']}"
    assert card["PAC"] >= 90
    assert card["DRI"] >= 90
    assert 45 <= card["DEF"] <= 99
    assert 45 <= card["PHY"] <= 99


def test_pitch_heatmap_coordinates(real_df):
    """Verifica que las coordenadas del mapa de calor se sitúan dentro de los límites del campo (105x68m)."""
    profiler = PlayerProfiler(real_df)
    yamal = real_df[real_df["player_name"] == "Lamine Yamal"].iloc[0]
    coords = profiler.get_pitch_heatmap_data(yamal, n_samples=100)

    assert len(coords) == 100
    assert coords["x"].min() >= 0.0 and coords["x"].max() <= 105.0
    assert coords["y"].min() >= 0.0 and coords["y"].max() <= 68.0


def test_player_twins_knn(real_df):
    """Verifica que el motor de clones encuentra futbolistas análogos con porcentaje de similitud coherente."""
    profiler = PlayerProfiler(real_df)
    mbappe = real_df[real_df["player_name"] == "Kylian Mbappé"].iloc[0]
    twins = profiler.find_player_twins(player_id=int(mbappe["player_id"]), top_k=3)

    assert len(twins) == 3
    assert twins.iloc[0]["similarity_pct"] >= 75.0
    assert "savings_vs_target_m" in twins.columns


def test_moneyball_bargains(real_df):
    """Verifica la detección de gangas por ratio de rendimiento / precio."""
    profiler = PlayerProfiler(real_df)
    bargains = profiler.calculate_moneyball_bargains(min_minutes=500, top_n=5)

    assert len(bargains) == 5
    assert bargains.iloc[0]["moneyball_ratio"] >= bargains.iloc[-1]["moneyball_ratio"]


def test_team_profiler_overview(real_df):
    """Comprueba que el perfil del club extrae su ADN táctico, Elo y alertas de veteranía."""
    elo = EloSystem()
    elo.ratings["Real Madrid"] = 1840.0
    profiler = TeamProfiler(df_players=real_df, elo_system=elo)

    overview = profiler.get_team_overview("Real Madrid")
    assert overview["team_name"] == "Real Madrid"
    assert overview["elo_rating"] == 1840.0
    assert "tactical_adn" in overview
    assert overview["tactical_adn"]["possession_pct"] > 50.0
    assert overview["total_market_value_m"] > 300.0
