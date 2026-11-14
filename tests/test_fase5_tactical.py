import pytest
import numpy as np
import pandas as pd
from src.models_tactical.lineup_simulator import TacticalSimulator
from src.models_tactical.matchup_analyzer import MatchupAnalyzer
from src.models_tactical.tactical_playmaker import TacticalPlaymaker
from src.models_tactical.tactical_analyst_desk import TacticalAnalystDesk
from src.models_analytics.poisson_model import DixonColesModel
from src.models_analytics.elo_model import EloSystem


@pytest.fixture
def mock_squad():
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "player_name": ["Extremo Titular", "Centrocampista", "Central Base", "Lateral Lento"],
        "position": ["FW", "MF", "DF", "DF"],
        "xg_p90": [0.15, 0.20, 0.02, 0.01],
        "xa_p90": [0.10, 0.35, 0.04, 0.08],
        "interceptions_p90": [0.4, 1.5, 2.2, 0.9],
        "tackles_p90": [0.3, 1.8, 1.9, 0.8],
        "is_reliable": [True, True, True, True],
    })


@pytest.fixture
def mock_elo():
    elo = EloSystem()
    elo.ratings["Manchester City"] = 1850.0
    elo.ratings["Sevilla FC"] = 1600.0
    elo.ratings["Club Modesto"] = 1400.0
    return elo


def test_player_swap_improves_metrics(mock_squad):
    """Verifica que fichar a un jugador con mejores métricas eleva lambda y los xPts."""
    sim = TacticalSimulator()
    top_candidate = {
        "xg_p90": 0.65,
        "xa_p90": 0.30,
        "interceptions_p90": 0.8,
    }

    result = sim.evaluate_player_swap(
        current_squad_df=mock_squad,
        player_out_id=1,
        candidate_stats=top_candidate,
        base_lambda=1.25,
        base_mu=1.10,
    )

    assert result["new_lambda"] > result["base_lambda"]
    assert result["new_xpts_per_match"] > result["base_xpts_per_match"]
    assert result["xpts_gain_38_matches"] > 0
    assert result["win_prob_delta"] > 0


def test_elo_context_adjustment(mock_squad, mock_elo):
    """Verifica que el factor de contexto Elo pondera el rendimiento según la liga/club de origen."""
    sim = TacticalSimulator(elo_system=mock_elo)
    candidate_stats = {"xg_p90": 0.40, "xa_p90": 0.20, "interceptions_p90": 0.5}

    # Jugador proveniente de club con Elo superior (City -> Sevilla)
    res_high = sim.evaluate_player_swap(
        current_squad_df=mock_squad,
        player_out_id=1,
        candidate_stats=candidate_stats,
        base_lambda=1.20,
        base_mu=1.10,
        candidate_origin_team="Manchester City",
        destination_team="Sevilla FC",
    )

    # Jugador de club con Elo inferior (Modesto -> Sevilla)
    res_low = sim.evaluate_player_swap(
        current_squad_df=mock_squad,
        player_out_id=1,
        candidate_stats=candidate_stats,
        base_lambda=1.20,
        base_mu=1.10,
        candidate_origin_team="Club Modesto",
        destination_team="Sevilla FC",
    )

    assert res_high["context_factor"] > res_low["context_factor"]
    assert res_high["new_lambda"] > res_low["new_lambda"]


def test_monte_carlo_simulation():
    """Comprueba que la simulación Monte Carlo proyecta medias coherentes de puntos."""
    sim = TacticalSimulator()
    mc_results = sim.simulate_monte_carlo_season(
        base_lambda=1.20,
        base_mu=1.15,
        new_lambda=1.55,
        new_mu=0.95,
        remaining_matches=10,
        current_points=45,
        n_simulations=500,
        seed=123,
    )

    assert mc_results["n_simulations"] == 500
    assert mc_results["new_projected_points_mean"] > mc_results["base_projected_points_mean"]
    assert mc_results["prob_gain_points"] > 0.50
    assert "new_prob_europe_60pts" in mc_results


def test_matchup_analyzer_optimal_lineup():
    """Verifica que el optimizador descarta jugadores no fiables y cubre la formación."""
    players = pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5],
        "player_name": ["Goleador Lesionado", "Delantero Fiable", "Medio Defensivo", "Central 1", "Central 2"],
        "position": ["FW", "FW", "MF", "DF", "DF"],
        "xg_p90": [0.95, 0.45, 0.10, 0.02, 0.03],
        "xa_p90": [0.40, 0.25, 0.30, 0.05, 0.02],
        "interceptions_p90": [0.1, 0.4, 2.1, 1.8, 1.9],
        "tackles_p90": [0.2, 0.5, 1.9, 1.5, 1.6],
        "is_reliable": [False, True, True, True, True],
    })

    analyzer = MatchupAnalyzer()
    formation = {"FW": 1, "MF": 1, "DF": 2}
    lineup = analyzer.recommend_optimal_lineup(
        available_players_df=players,
        formation=formation,
        opponent_profile={"defense_weakness": 1.25, "attack_threat": 0.85},
    )

    assert len(lineup) == 4
    # Jugador no fiable no debe estar en la alineación
    assert 1 not in lineup["player_id"].values
    assert 2 in lineup["player_id"].values


def test_tactical_playmaker_rules():
    """Verifica la generación de jugadas tácticas según perfiles estadísticos."""
    playmaker = TacticalPlaymaker()

    # Once con alta finalización frente a defensa rival adelantada
    lineup = pd.DataFrame({
        "xg_p90": [0.40, 0.35, 0.25],
        "xa_p90": [0.10, 0.15, 0.10],
        "interceptions_p90": [0.5, 0.6, 0.8],
        "tackles_p90": [0.4, 0.5, 0.6],
    })

    opponent = {
        "high_line_vulnerability": 0.75,
        "low_block_tendency": 0.20,
        "attack_threat": 1.0,
    }

    plays = playmaker.suggest_tactical_plays(lineup, opponent)
    categories = [p["categoria"] for p in plays]

    assert "Transición Ofensiva" in categories
    assert any("Transición Vertical" in p["jugada"] for p in plays)


def test_tactical_analyst_desk_end_to_end():
    """Prueba de integración del Virtual Analyst Desk conectando diagnóstico, scouting, simulación y plan de partido."""
    df_players = pd.DataFrame({
        "player_id": [101, 102, 103, 104, 201, 202],
        "player_name": ["Sevilla FW Flojo", "Sevilla MF Bueno", "Sevilla DF Sólido", "Sevilla GK", "Fichaje Premier FW", "Fichaje Liga Portugal FW"],
        "team_name": ["Sevilla FC", "Sevilla FC", "Sevilla FC", "Sevilla FC", "Arsenal", "Benfica"],
        "league": ["LaLiga", "LaLiga", "LaLiga", "LaLiga", "Premier League", "Liga Portugal"],
        "position": ["FW", "MF", "DF", "GK", "FW", "FW"],
        "age": [29, 25, 27, 28, 23, 22],
        "minutes": [1800, 1600, 2100, 1900, 1500, 1400],
        "xg_p90": [0.12, 0.22, 0.02, 0.00, 0.58, 0.48],
        "xa_p90": [0.08, 0.32, 0.04, 0.00, 0.25, 0.22],
        "interceptions_p90": [0.3, 1.4, 2.3, 0.1, 0.7, 0.6],
        "tackles_p90": [0.2, 1.6, 1.8, 0.1, 0.6, 0.5],
        "is_reliable": [True, True, True, True, True, True],
    })

    elo = EloSystem()
    elo.ratings["Arsenal"] = 1820.0
    elo.ratings["Benfica"] = 1680.0
    elo.ratings["Sevilla FC"] = 1600.0
    elo.ratings["Real Betis"] = 1610.0

    desk = TacticalAnalystDesk(df_players_metrics=df_players, elo_system=elo)

    # 1. Auditoría de Sevilla FC
    audit = desk.audit_club_squad(team_name="Sevilla FC", metric="xg_p90")
    assert audit["team_name"] == "Sevilla FC"
    assert "deficits_table" in audit

    # 2. Scouting en Premier League y Liga Portugal
    targets = desk.scout_transfer_targets(
        target_position="FW",
        leagues_filter=["Premier League", "Liga Portugal"],
        top_k=2
    )
    assert len(targets) == 2
    assert "Fichaje Premier FW" in targets["player_name"].values

    # 3. Simular fichaje "What-If" del delantero de la Premier en lugar del titular
    transfer_sim = desk.simulate_transfer_scenario(
        team_name="Sevilla FC",
        player_out_id=101,
        candidate_player_id=201,
        base_lambda=1.20,
        base_mu=1.10,
        remaining_matches=10,
        current_points=40,
    )
    assert transfer_sim["swap_impact"]["new_lambda"] > transfer_sim["swap_impact"]["base_lambda"]
    assert transfer_sim["swap_impact"]["xpts_gain_per_match"] > 0
    assert "monte_carlo_projection" in transfer_sim

    # 4. Preparar plan de partido vs Real Betis
    match_plan = desk.prepare_match_plan(
        team_name="Sevilla FC",
        opponent_profile={"defense_weakness": 1.2, "attack_threat": 1.1, "high_line_vulnerability": 0.7},
        formation={"GK": 1, "DF": 1, "MF": 1, "FW": 1}
    )
    assert len(match_plan["optimal_lineup"]) == 4
    assert len(match_plan["tactical_plays"]) >= 1

