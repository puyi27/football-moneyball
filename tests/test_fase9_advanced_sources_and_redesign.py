import pytest
import pandas as pd
import numpy as np
from src.data.advanced_sources_hub import (
    MasterAnalyticsDataHub,
    StatsBombEngine,
    UnderstatShotEngine,
    SoccerDataHub,
    KloppyStandardizer,
    OpenFootballFeed,
    TransfermarktDatasetHub
)

def test_master_hub_health_check():
    hub = MasterAnalyticsDataHub()
    health = hub.get_system_health()
    assert isinstance(health, dict)
    assert "soccerdata" in health
    assert "statsbombpy" in health
    assert "kloppy" in health
    assert "openfootball" in health
    assert "understat" in health
    assert "transfermarkt" in health
    assert "football_data" in health

def test_statsbomb_engine_coordinates():
    sb_engine = StatsBombEngine()
    comps = sb_engine.get_featured_competitions()
    assert not comps.empty
    assert "competition_id" in comps.columns

    events = sb_engine.get_match_events(match_id=22912)
    assert not events.empty
    assert "location_x" in events.columns
    assert "location_y" in events.columns
    assert "type" in events.columns

def test_understat_shot_matrix_and_efficiency():
    understat = UnderstatShotEngine()
    shots = understat.get_shot_data(limit=100)
    assert not shots.empty
    assert "xg" in shots.columns
    assert "result" in shots.columns
    assert "x" in shots.columns
    assert "y" in shots.columns
    assert "player" in shots.columns

    eff = understat.compute_player_finishing_efficiency(shots)
    assert not eff.empty
    assert "goals_minus_xg" in eff.columns
    assert "finishing_ratio" in eff.columns
    assert "is_elite_finisher" in eff.columns

def test_kloppy_pysport_standardization():
    kloppy = KloppyStandardizer()
    raw_df = pd.DataFrame([
        {"type": "Pass", "team": "Sevilla FC", "location_x": 52.5, "location_y": 34.0, "minute": 12, "second": 30},
        {"type": "Shot", "team": "Sevilla FC", "location_x": 95.0, "location_y": 35.0, "minute": 15, "second": 10},
        {"type": "Duel", "team": "Real Madrid", "location_x": 40.0, "location_y": 20.0, "minute": 20, "second": 0}
    ])
    std = kloppy.standardize_event_stream(raw_df)
    assert len(std) == 3
    assert "norm_x" in std.columns
    assert "norm_y" in std.columns
    assert "action_type" in std.columns
    assert std.iloc[0]["action_type"] == "PASS"
    assert std.iloc[1]["action_type"] == "SHOT"
    assert 0 <= std.iloc[0]["norm_x"] <= 100
    assert 0 <= std.iloc[0]["norm_y"] <= 100

def test_openfootball_feed():
    feed = OpenFootballFeed()
    matches = feed.get_league_season_matches()
    assert not matches.empty
    assert "team1" in matches.columns
    assert "team2" in matches.columns

def test_transfermarkt_moneyball_contract_bargains():
    tm = TransfermarktDatasetHub()
    opps = tm.get_moneyball_contract_opportunities()
    assert not opps.empty
    assert "player_name" in opps.columns
    assert "market_value_m" in opps.columns
    assert "months_left" in opps.columns
    assert "opportunity_score" in opps.columns
    # Check that short contracts offer expected savings
    top = opps.iloc[0]
    assert top["months_left"] <= 12 or top["opportunity_score"] > 80

def test_soccerdata_fbref_advanced_p90():
    sd_hub = SoccerDataHub()
    stats = sd_hub.get_advanced_p90_metrics()
    assert not stats.empty
    assert "player" in stats.columns
    assert "xg_p90" in stats.columns
    assert "xag_p90" in stats.columns
    assert "sca_p90" in stats.columns
