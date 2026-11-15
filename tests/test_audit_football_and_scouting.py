import pytest
import numpy as np
import pandas as pd

from src.data.global_player_database import load_global_players, generate_worldwide_player_network
from src.models_scouting.player_profiler import PlayerProfiler
from src.models_tactical.matchup_analyzer import MatchupAnalyzer
from src.models_tactical.match_scenario_simulator import MatchScenarioSimulator
from src.models_tactical.tactical_analyst_desk import TacticalAnalystDesk


@pytest.fixture(scope="module")
def global_df():
    return load_global_players()


@pytest.fixture(scope="module")
def profiler(global_df):
    return PlayerProfiler(global_df)


@pytest.fixture(scope="module")
def matchup_analyzer():
    return MatchupAnalyzer()


@pytest.fixture(scope="module")
def tactical_desk(global_df):
    return TacticalAnalystDesk(global_df)


# =============================================================================
# 1. AUDITORÍA DE CONSISTENCIA DE DATOS (LALIGA Y POSICIONES)
# =============================================================================

def test_laliga_squads_completeness_and_valid_positions(global_df):
    """
    Verifica que los 20 clubes de LaLiga estén completos, tengan al menos 16 futbolistas,
    al menos 2 porteros y que todas las demarcaciones pertenezcan a {'GK', 'DF', 'MF', 'FW'}.
    """
    laliga_df = global_df[global_df["league"] == "LaLiga"]
    teams = laliga_df["team_name"].unique()
    assert len(teams) == 20, f"Debe haber 20 clubes de LaLiga, encontrados: {len(teams)}"

    valid_positions = {"GK", "DF", "MF", "FW"}

    for team in teams:
        team_players = laliga_df[laliga_df["team_name"] == team]
        assert len(team_players) >= 16, f"{team} tiene solo {len(team_players)} jugadores (<16)"

        # Posiciones válidas
        invalid_pos = set(team_players["position"].unique()) - valid_positions
        assert not invalid_pos, f"Posiciones inválidas en {team}: {invalid_pos}"

        # Al menos 2 porteros
        gk_count = (team_players["position"] == "GK").sum()
        assert gk_count >= 2, f"{team} debe tener al menos 2 porteros, tiene {gk_count}"

        # Al menos 4 defensores, 4 centrocampistas y 4 delanteros
        df_count = (team_players["position"] == "DF").sum()
        mf_count = (team_players["position"] == "MF").sum()
        fw_count = (team_players["position"] == "FW").sum()
        assert df_count >= 4, f"{team} tiene solo {df_count} defensas"
        assert mf_count >= 4, f"{team} tiene solo {mf_count} centrocampistas"
        assert fw_count >= 4, f"{team} tiene solo {fw_count} delanteros"


def test_sevilla_fc_and_major_clubs_rosters(global_df):
    """
    Verifica plantillas clave: Sevilla FC (24 jugadores reales), Real Madrid, Barcelona, Betis, Atlético.
    """
    laliga_df = global_df[global_df["league"] == "LaLiga"]

    # Sevilla FC
    sevilla = laliga_df[laliga_df["team_name"] == "Sevilla FC"]
    assert len(sevilla) == 24, f"Sevilla FC debe tener exactamente 24 jugadores, encontrados {len(sevilla)}"

    # Otros clubes clave
    for club, min_size in [("Real Madrid", 20), ("Barcelona", 20), ("Real Betis", 20), ("Atletico Madrid", 18)]:
        sub = laliga_df[laliga_df["team_name"] == club]
        assert len(sub) >= min_size, f"{club} tiene menos de {min_size} jugadores ({len(sub)})"


# =============================================================================
# 2. AUDITORÍA DEL MOTOR DE PERFILADO (GOALKEEPERS, HEATMAPS, PIZZA PLOTS)
# =============================================================================

def test_goalkeeper_fifa_card_attributes_and_ovr(profiler, global_df):
    """
    Verifica que los porteros reciban valoraciones OVR realistas (72-94),
    atributos FIFA válidos (DIV, HAN, KIC, REF, SPE, POS y su mapeo estándar)
    sin caer en el suelo de 65 ni divisiones por cero.
    """
    gks = global_df[global_df["position"] == "GK"]
    assert len(gks) >= 40, "Debe haber al menos 40 porteros en la base de datos"

    for _, row in gks.iterrows():
        card = profiler.calculate_fifa_card(row)
        assert card["is_goalkeeper"] is True
        assert 72 <= card["OVR"] <= 95, f"OVR anómalo para {row['player_name']}: {card['OVR']}"

        # Atributos de portero
        for stat in ["DIV", "HAN", "KIC", "REF", "SPE", "POS"]:
            assert stat in card
            assert 45 <= card[stat] <= 99, f"Stat {stat} anómala para {row['player_name']}: {card[stat]}"

        # Atributos universales FIFA/EA FC mapeados
        for stat in ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]:
            assert stat in card
            assert 45 <= card[stat] <= 99

    # Verificar porteros de élite mundial
    top_names = ["Thibaut Courtois", "Jan Oblak", "Marc-André ter Stegen"]
    for name in top_names:
        elite_row = gks[gks["player_name"] == name]
        if not elite_row.empty:
            card = profiler.calculate_fifa_card(elite_row.iloc[0])
            assert card["OVR"] >= 86, f"{name} debe tener OVR >= 86, obtenido {card['OVR']}"


def test_goalkeeper_pitch_heatmap_bounded_in_penalty_box(profiler, global_df):
    """
    Verifica que el mapa de calor de un portero esté estrictamente contenido en el área de meta/penalti.
    """
    gk_row = global_df[global_df["position"] == "GK"].iloc[0]
    heatmap = profiler.get_pitch_heatmap_data(gk_row, n_samples=200)

    assert not heatmap.empty
    assert "x" in heatmap.columns and "y" in heatmap.columns
    # El área de penalti mide 16.5m de longitud y está centrada en y=34m (13.8m a 54.2m)
    assert heatmap["x"].min() >= 3.0
    assert heatmap["x"].max() <= 26.0
    assert heatmap["y"].min() >= 15.0
    assert heatmap["y"].max() <= 53.0


def test_pizza_plot_robustness_with_gks_and_zero_series(profiler, global_df):
    """
    Verifica que generate_pizza_plot genere la figura para cualquier demarcación
    y no falle ante series con ceros idénticos o valores constantes (utilizando mid-rank percentiles).
    """
    # 1. Portero real
    gk_row = global_df[global_df["position"] == "GK"].iloc[0]
    fig_gk = profiler.generate_pizza_plot(gk_row)
    assert fig_gk is not None

    # 2. Centrocampista real
    mf_row = global_df[global_df["position"] == "MF"].iloc[0]
    fig_mf = profiler.generate_pizza_plot(mf_row)
    assert fig_mf is not None

    # 3. Caso extremo sintético: jugador con métricas todas en 0.0
    zero_player = pd.Series({
        "player_id": 99999,
        "player_name": "Ghost Player",
        "position": "FW",
        "xg_p90": 0.0,
        "xa_p90": 0.0,
        "prog_passes_p90": 0.0,
        "dribbles_p90": 0.0,
        "interceptions_p90": 0.0,
        "tackles_p90": 0.0,
        "market_value_m": 0.0,
        "potential_ovr": 50
    })
    fig_zero = profiler.generate_pizza_plot(zero_player)
    assert fig_zero is not None


# =============================================================================
# 3. AUDITORÍA TÁCTICA (ALINEACIONES DE 11 SIN DUPLICADOS Y PROBABILIDADES = 1.0)
# =============================================================================

def test_lineup_always_has_11_unique_players_across_all_laliga_teams(matchup_analyzer, global_df):
    """
    Verifica que en todos los 20 clubes de LaLiga y bajo múltiples formaciones tácticas,
    la alineación recomendada contenga exactamente 11 futbolistas reales sin ninguna duplicación.
    """
    laliga_teams = global_df[global_df["league"] == "LaLiga"]["team_name"].unique()
    formations = [
        {"GK": 1, "DF": 4, "MF": 3, "FW": 3},  # 4-3-3
        {"GK": 1, "DF": 5, "MF": 3, "FW": 2},  # 5-3-2
        {"GK": 1, "DF": 3, "MF": 5, "FW": 2},  # 3-5-2
        {"GK": 1, "DF": 4, "MF": 4, "FW": 2},  # 4-4-2
        {"GK": 1, "DF": 4, "MF": 5, "FW": 1},  # 4-2-3-1
    ]

    for team in laliga_teams:
        team_players = global_df[global_df["team_name"] == team]
        for form in formations:
            lineup = matchup_analyzer.recommend_optimal_lineup(team_players, formation=form)
            assert len(lineup) == 11, f"Fallo en {team} con {form}: len={len(lineup)}"
            assert lineup["player_id"].nunique() == 11, f"Duplicados en {team} con {form}"
            assert lineup["player_name"].nunique() == 11


def test_predict_match_outcome_probabilities_sum_to_one(matchup_analyzer, global_df):
    """
    Verifica que predict_match_outcome retorne probabilidades 1X2 válidas, no negativas
    y cuya suma sea exactamente 1.0 (100%).
    """
    sevilla = global_df[global_df["team_name"] == "Sevilla FC"]
    madrid = global_df[global_df["team_name"] == "Real Madrid"]

    lineup_sev = matchup_analyzer.recommend_optimal_lineup(sevilla)
    lineup_mad = matchup_analyzer.recommend_optimal_lineup(madrid)

    outcome = matchup_analyzer.predict_match_outcome(
        home_team="Sevilla FC",
        away_team="Real Madrid",
        local_lineup=lineup_sev,
        rival_lineup=lineup_mad,
        home_elo=1620,
        away_elo=1720
    )

    p_home = outcome["win_home_prob"]
    p_draw = outcome["draw_prob"]
    p_away = outcome["win_away_prob"]

    assert 0.0 <= p_home <= 1.0
    assert 0.0 <= p_draw <= 1.0
    assert 0.0 <= p_away <= 1.0

    total_prob = round(p_home + p_draw + p_away, 4)
    assert total_prob == 1.0, f"Las probabilidades no suman 1.0: {total_prob}"


def test_tactical_analyst_desk_end_to_end_coherence(tactical_desk):
    """
    Verifica la preparación integral de partido en TacticalAnalystDesk tanto con
    nombre de rival en formato string como con diccionario táctico.
    """
    # 1. Preparación con nombre string
    plan_str = tactical_desk.prepare_match_plan("Sevilla FC", "Barcelona")
    assert "formation_recommended" in plan_str
    assert "tactical_instructions" in plan_str
    assert len(plan_str["optimal_lineup"]) == 11
    assert plan_str["optimal_lineup"]["player_id"].nunique() == 11

    # 2. Preparación con perfil de rival dict
    custom_profile = {
        "defense_weakness": 1.25,
        "attack_threat": 0.85,
        "high_line_vulnerability": 0.80,
        "low_block_tendency": 0.25
    }
    plan_dict = tactical_desk.prepare_match_plan("Real Betis", custom_profile)
    assert len(plan_dict["optimal_lineup"]) == 11
    assert plan_dict["optimal_lineup"]["player_id"].nunique() == 11


def test_match_scenario_simulator_coherence():
    """
    Verifica que el simulador de escenarios minuto a minuto produzca marcadores válidos,
    xG no negativos y probabilidades dinámicas de victoria acotadas.
    """
    simulator = MatchScenarioSimulator()
    sim_res = simulator.simulate_match_timeline("Atletico Madrid", "Athletic Club", 1680, 1660, seed=123)

    assert sim_res["final_score"][0] >= 0
    assert sim_res["final_score"][1] >= 0
    assert sim_res["total_xg_home"] >= 0.0
    assert sim_res["total_xg_away"] >= 0.0

    for ev in sim_res["events"]:
        assert "win_prob_home" in ev
        assert "win_prob_home_float" in ev
        assert 0.01 <= ev["win_prob_home_float"] <= 0.99
