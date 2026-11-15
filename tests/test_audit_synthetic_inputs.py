import pytest
import numpy as np
import pandas as pd

from src.data.global_player_database import load_global_players
from src.data.real_dataset_loader import load_real_players_df
from src.data.football_api_connector import FootballAPIConnector
from src.data.advanced_sources_hub import MasterAnalyticsDataHub

from src.models_scouting.player_profiler import PlayerProfiler
from src.models_scouting.recommender import ScoutingRecommender
from src.models_scouting.squad_diagnostics import SquadDiagnostics
from src.models_scouting.academy_scout import AcademyScout

from src.models_analytics.elo_model import EloSystem
from src.models_analytics.poisson_model import DixonColesModel
from src.models_analytics.team_profiler import TeamProfiler
from src.models_analytics.evaluation import ModelEvaluator

from src.models_tactical.lineup_simulator import TacticalSimulator
from src.models_tactical.matchup_analyzer import MatchupAnalyzer
from src.models_tactical.tactical_playmaker import TacticalPlaymaker
from src.models_tactical.tactical_analyst_desk import TacticalAnalystDesk
from src.models_tactical.match_scenario_simulator import MatchScenarioSimulator


@pytest.fixture(scope="module")
def global_df():
    return load_global_players()


@pytest.fixture
def synthetic_players_df(global_df):
    """
    Conjunto sintético exhaustivo de prueba:
    1. Delantero estrella de campo
    2. Portero consagrado (GK)
    3. Canterano joven (17 años, Wonderkid)
    4. Jugador con 0 minutos jugados en la temporada
    5. Jugador con métricas NaN / None
    """
    rows = [
        # 1. Delantero de campo
        {
            "player_id": 9001, "player_name": "Field Striker", "team_name": "Test FC", "league": "LaLiga",
            "position": "FW", "age": 24, "market_value_m": 80.0, "minutes": 2200, "xg_p90": 0.65, "xa_p90": 0.30,
            "goals_p90": 0.70, "assists_p90": 0.25, "interceptions_p90": 0.40, "tackles_p90": 0.60,
            "dribbles_p90": 3.20, "prog_passes_p90": 4.50, "potential_ovr": 91, "is_academy": False,
            "academy_origin": "Test Academy", "is_reliable": True
        },
        # 2. Portero (GK)
        {
            "player_id": 9002, "player_name": "Test Goalkeeper", "team_name": "Test FC", "league": "LaLiga",
            "position": "GK", "age": 31, "market_value_m": 25.0, "minutes": 2500, "xg_p90": 0.00, "xa_p90": 0.00,
            "goals_p90": 0.00, "assists_p90": 0.00, "interceptions_p90": 0.20, "tackles_p90": 0.15,
            "dribbles_p90": 0.05, "prog_passes_p90": 1.20, "potential_ovr": 86, "is_academy": False,
            "academy_origin": "GK Academy", "is_reliable": True
        },
        # 3. Canterano Wonderkid
        {
            "player_id": 9003, "player_name": "Academy Gem", "team_name": "Test FC", "league": "LaLiga",
            "position": "MF", "age": 17, "market_value_m": 12.0, "minutes": 850, "xg_p90": 0.22, "xa_p90": 0.45,
            "goals_p90": 0.18, "assists_p90": 0.40, "interceptions_p90": 1.40, "tackles_p90": 1.80,
            "dribbles_p90": 2.80, "prog_passes_p90": 6.80, "potential_ovr": 92, "is_academy": True,
            "academy_origin": "Cantera Test FC", "is_reliable": True
        },
        # 4. Jugador con 0 minutos
        {
            "player_id": 9004, "player_name": "Zero Minutes Sub", "team_name": "Test FC", "league": "LaLiga",
            "position": "DF", "age": 20, "market_value_m": 1.5, "minutes": 0, "xg_p90": 0.00, "xa_p90": 0.00,
            "goals_p90": 0.00, "assists_p90": 0.00, "interceptions_p90": 0.00, "tackles_p90": 0.00,
            "dribbles_p90": 0.00, "prog_passes_p90": 0.00, "potential_ovr": 78, "is_academy": True,
            "academy_origin": "Cantera Test FC", "is_reliable": False
        },
        # 5. Jugador con NaN / None
        {
            "player_id": 9005, "player_name": "Null Metrics Player", "team_name": "Test FC", "league": "LaLiga",
            "position": "DF", "age": np.nan, "market_value_m": np.nan, "minutes": np.nan, "xg_p90": np.nan,
            "xa_p90": np.nan, "goals_p90": np.nan, "assists_p90": np.nan, "interceptions_p90": np.nan,
            "tackles_p90": np.nan, "dribbles_p90": np.nan, "prog_passes_p90": np.nan, "potential_ovr": np.nan,
            "is_academy": None, "academy_origin": None, "is_reliable": None
        }
    ]
    synth_df = pd.DataFrame(rows)
    # Concatenar con una muestra del global para asegurar riqueza de posiciones y K-NN
    return pd.concat([synth_df, global_df.head(40)], ignore_index=True)


# =============================================================================
# 1. PRUEBAS SINTÉTICAS: PLAYER PROFILER
# =============================================================================

def test_player_profiler_with_all_archetypes(synthetic_players_df):
    """
    Verifica que PlayerProfiler procesa sin errores jugadores de campo, porteros,
    canteranos, jugadores con 0 minutos y jugadores con nulos.
    """
    profiler = PlayerProfiler(synthetic_players_df)

    for _, player in synthetic_players_df.iterrows():
        # 1. Cálculo de tarjeta FIFA
        card = profiler.calculate_fifa_card(player)
        assert isinstance(card, dict)
        assert 45 <= card["OVR"] <= 99
        assert 45 <= card["PAC"] <= 99
        assert 45 <= card["SHO"] <= 99
        assert 45 <= card["PAS"] <= 99
        assert 45 <= card["DRI"] <= 99
        assert 45 <= card["DEF"] <= 99
        assert 45 <= card["PHY"] <= 99
        assert isinstance(card["age"], int)
        assert isinstance(card["market_value_m"], float)

        # 2. Generación de mapa de calor 2D
        hm = profiler.get_pitch_heatmap_data(player, n_samples=30)
        assert len(hm) == 30
        assert 0.0 <= hm["x"].min() and hm["x"].max() <= 105.0
        assert 0.0 <= hm["y"].min() and hm["y"].max() <= 68.0

        # Comprobar alias
        hm_alias = profiler.generate_pitch_heatmap(player, n_samples=20)
        assert len(hm_alias) == 20

    # 3. Comprobar find_player_twins con firmas combinadas
    # A) Por player_id (int) y top_k (int)
    twins_id = profiler.find_player_twins(player_id=9001, top_k=3)
    assert not twins_id.empty
    assert "similarity_pct" in twins_id.columns
    assert "similarity_score" in twins_id.columns
    assert "potential_ovr" in twins_id.columns

    # B) Por player_name (str) y n_neighbors (int) - según llamada de app.py
    twins_name = profiler.find_player_twins("Field Striker", n_neighbors=3)
    assert not twins_name.empty
    assert len(twins_name) == 3

    # C) Con portero (GK)
    twins_gk = profiler.find_player_twins("Test Goalkeeper", n_neighbors=2)
    assert isinstance(twins_gk, pd.DataFrame)

    # D) Con jugador de 0 minutos
    twins_zero = profiler.find_player_twins(player_id=9004, top_k=2)
    assert isinstance(twins_zero, pd.DataFrame)

    # 4. Gangas Moneyball con 0 minutos y con umbral alto
    bargains_0 = profiler.calculate_moneyball_bargains(min_minutes=0, top_n=5)
    assert len(bargains_0) <= 5
    assert not bargains_0.empty

    bargains_high = profiler.calculate_moneyball_bargains(min_minutes=3500, top_n=5)
    assert isinstance(bargains_high, pd.DataFrame)


# =============================================================================
# 2. PRUEBAS SINTÉTICAS: SCOUTING RECOMMENDER
# =============================================================================

def test_scouting_recommender_with_all_archetypes(synthetic_players_df):
    """
    Verifica que ScoutingRecommender maneja ordenamiento ponderado y K-NN
    con jugadores fiables, no fiables (0 minutos) y diferentes tipos de entradas.
    """
    recommender = ScoutingRecommender(synthetic_players_df)

    # 1. Puntuación ponderada
    weights = {"xg_p90": 1.5, "interceptions_p90": 0.8, "goals_p90": 1.0}
    scored = recommender.calculate_weighted_score(weights)
    assert not scored.empty
    assert "custom_score" in scored.columns
    assert scored["custom_score"].iloc[0] >= scored["custom_score"].iloc[-1]

    # 2. Búsqueda de similares por player_id (int)
    sims_id = recommender.find_similar_players(player_id=9001, features=["xg_p90", "xa_p90"], k=2)
    assert len(sims_id) == 2
    assert "similarity_distance" in sims_id.columns

    # 3. Búsqueda de similares por player_name (str)
    sims_name = recommender.find_similar_players(player_id="Field Striker", features=["xg_p90", "xa_p90"], k=2)
    assert len(sims_name) == 2


# =============================================================================
# 3. PRUEBAS SINTÉTICAS: SQUAD DIAGNOSTICS
# =============================================================================

def test_squad_diagnostics_with_all_archetypes(synthetic_players_df):
    """
    Verifica que SquadDiagnostics opera tanto con string de club como con DataFrame,
    e itera .items() como requiere app.py.
    """
    diag = SquadDiagnostics(synthetic_players_df)

    # 1. Entrada como string
    deficits_str = diag.analyze_team_deficits("Test FC", metric="xg_p90")
    assert not deficits_str.empty
    assert "needs_reinforcement" in deficits_str.columns
    assert "deficit" in deficits_str.columns

    # 2. Comprobar que .items() funciona exactamente como app.py lo consume
    count_items = 0
    for pos, info in deficits_str.items():
        count_items += 1
        assert isinstance(pos, str)
        assert isinstance(info, dict)
        assert info["status"] in ["CRITICAL", "WARNING", "OK", "BALANCED"]
        assert "reason" in info

    assert count_items > 0

    # 3. Entrada como DataFrame (squad)
    squad_subset = synthetic_players_df[synthetic_players_df["team_name"] == "Test FC"]
    deficits_df = diag.analyze_team_deficits(squad_subset, metric="xg_p90")
    assert not deficits_df.empty

    # 4. Alertas de riesgo por veteranía y minutos
    alerts = diag.detect_squad_risk_alerts("Test FC", max_age=30, min_minutes_share=0.1)
    assert isinstance(alerts, pd.DataFrame)

    alerts_df_in = diag.detect_squad_risk_alerts(squad_subset, max_age=30, min_minutes_share=0.1)
    assert isinstance(alerts_df_in, pd.DataFrame)


# =============================================================================
# 4. PRUEBAS SINTÉTICAS: ACADEMY SCOUT
# =============================================================================

def test_academy_scout_with_all_archetypes(synthetic_players_df):
    """
    Verifica detección de canteranos, proyección de carrera y comparador cantera vs mercado.
    """
    scout = AcademyScout(synthetic_players_df)

    # 1. Prospección de promesas
    prospects = scout.get_academy_prospects(team_name="Test FC", min_potential=75, max_age=23)
    assert not prospects.empty
    assert "projected_savings_m" in prospects.columns

    # 2. Pathway por nombre (str) y por ID (int)
    path_name = scout.recommend_career_pathway("Academy Gem", target_team="Test FC")
    assert path_name["player_name"] == "Academy Gem"
    assert "recommended_pathway" in path_name
    assert path_name["readiness_score_pct"] > 0

    path_id = scout.recommend_career_pathway(9003, target_team="Test FC")
    assert path_id["player_name"] == "Academy Gem"

    # 3. Comparativa cantera vs mercado (str y alias compare_academy_vs_market)
    comp = scout.compare_academy_vs_market("Academy Gem", "Field Striker")
    assert "academy_prospect" in comp
    assert "senior_star" in comp
    assert "financial_verdict" in comp
    assert comp["financial_verdict"]["net_savings_eur_m"] > 0

    # 4. Comparativa con IDs numéricos
    comp_ids = scout.compare_academy_vs_market(9003, 9001)
    assert "financial_verdict" in comp_ids


# =============================================================================
# 5. PRUEBAS SINTÉTICAS: TEAM PROFILER & ELO SYSTEM
# =============================================================================

def test_team_profiler_with_all_archetypes(synthetic_players_df):
    """
    Verifica que TeamProfiler genera el reporte completo con todas las claves esperadas por app.py.
    """
    elo = EloSystem()
    elo.ratings["Test FC"] = 1750.0
    profiler = TeamProfiler(synthetic_players_df, elo_system=elo)

    overview = profiler.get_team_overview("Test FC")
    assert overview["team_name"] == "Test FC"
    assert overview["squad_size"] == len(synthetic_players_df[synthetic_players_df["team_name"] == "Test FC"])
    assert overview["total_squad_size"] == overview["squad_size"]
    assert overview["avg_age"] > 0
    assert overview["avg_squad_age"] == overview["avg_age"]
    assert "academy_players_count" in overview
    assert "tactical_adn" in overview
    assert "aging_renewals_needed" in overview


# =============================================================================
# 6. PRUEBAS SINTÉTICAS: TACTICAL SIMULATOR, MATCHUP ANALYZER & PLAYMAKER
# =============================================================================

def test_tactical_suite_with_all_archetypes(synthetic_players_df):
    """
    Verifica el funcionamiento coordinado de TacticalSimulator, MatchupAnalyzer y TacticalPlaymaker.
    """
    sim = TacticalSimulator()
    analyzer = MatchupAnalyzer()
    playmaker = TacticalPlaymaker()

    squad = synthetic_players_df[synthetic_players_df["team_name"] == "Test FC"]

    # 1. Matchup Analyzer: Tactical Fit
    fit_fw = analyzer.calculate_player_tactical_fit(squad.iloc[0], {"defense_weakness": 1.2, "attack_threat": 1.1})
    assert isinstance(fit_fw, float)

    fit_gk = analyzer.calculate_player_tactical_fit(squad.iloc[1], {"defense_weakness": 1.0, "attack_threat": 1.0})
    assert isinstance(fit_gk, float)

    fit_null = analyzer.calculate_player_tactical_fit(squad.iloc[4], {"defense_weakness": 1.0, "attack_threat": 1.0})
    assert isinstance(fit_null, float)

    # 2. Recomendación de alineación
    lineup = analyzer.recommend_optimal_lineup(
        available_players_df=squad,
        formation={"FW": 1, "MF": 1, "DF": 1},
        require_reliable=False
    )
    assert len(lineup) == 3

    # 3. Predicción de partido 1X2 (utilizado en app.py)
    pred_1x2 = analyzer.predict_match_outcome(
        home_team="Test FC",
        away_team="Rival FC",
        local_lineup=lineup,
        rival_lineup=lineup
    )
    assert "win_home_prob" in pred_1x2
    assert "draw_prob" in pred_1x2
    assert "win_away_prob" in pred_1x2
    total_prob = pred_1x2["win_home_prob"] + pred_1x2["draw_prob"] + pred_1x2["win_away_prob"]
    assert total_prob == pytest.approx(1.0, abs=1e-2)

    # 4. Jugadas recomendadas
    plays = playmaker.suggest_tactical_plays(
        lineup_df=lineup,
        opponent_profile={"high_line_vulnerability": 0.8, "low_block_tendency": 0.3, "attack_threat": 1.2}
    )
    assert len(plays) >= 1

    # 5. Intercambio de jugadores What-If
    swap_res = sim.evaluate_player_swap(
        current_squad_df=squad,
        player_out_id=9001,
        candidate_stats={"xg_p90": 0.85, "xa_p90": 0.35, "interceptions_p90": 0.50},
        base_lambda=1.30,
        base_mu=1.10
    )
    assert swap_res["new_lambda"] > swap_res["base_lambda"]
    assert "win_prob_delta" in swap_res


# =============================================================================
# 7. PRUEBAS SINTÉTICAS: DIGITAL TWIN & VIRTUAL ANALYST DESK
# =============================================================================

def test_digital_twin_and_analyst_desk(synthetic_players_df):
    """
    Verifica que MatchScenarioSimulator y TacticalAnalystDesk responden
    a simulaciones estocásticas y planes de partido de extremo a extremo.
    """
    # 1. Digital Twin
    scenario_sim = MatchScenarioSimulator()
    sim_out = scenario_sim.simulate_match_timeline("Real Madrid", "Barcelona", base_elo_home=1840, base_elo_away=1820, seed=42)
    assert "final_score" in sim_out
    assert "total_xg_home" in sim_out
    assert "events" in sim_out
    assert len(sim_out["events"]) > 0

    # 2. Virtual Analyst Desk
    desk = TacticalAnalystDesk(df_players_metrics=synthetic_players_df)
    audit = desk.audit_club_squad(team_name="Test FC", metric="xg_p90")
    assert audit["team_name"] == "Test FC"

    plan = desk.prepare_match_plan("Test FC", "Barcelona")
    assert "formation_recommended" in plan
    assert "tactical_instructions" in plan
    assert len(plan["optimal_lineup"]) > 0
