import pytest
import pandas as pd
from src.data.real_dataset_loader import load_real_players_df
from src.models_tactical.match_scenario_simulator import MatchScenarioSimulator
from src.models_scouting.academy_scout import AcademyScout


@pytest.fixture
def real_df():
    return load_real_players_df()


def test_match_scenario_simulator_execution():
    """Verifica que la simulación minuto a minuto genera eventos, línea temporal de xG y marcador coherente."""
    sim = MatchScenarioSimulator()
    res = sim.simulate_situational_match(
        home_team="Real Madrid",
        away_team="Barcelona",
        coach_formation="4-3-3",
        coach_style="Posesión y Bloque Alto",
        coach_pressing=8.0,
        seed=42
    )

    assert "coach_tactics" in res
    assert "ai_counter_tactics" in res
    assert "own_vulnerabilities_alert" in res

    coach = res["coach_tactics"]
    assert coach["expected_lambda"] > 0
    assert coach["expected_mu"] > 0
    assert len(coach["xg_timeline"]) == 90
    assert isinstance(coach["events_log"], list)
    assert "-" in coach["simulated_score"]

    ai = res["ai_counter_tactics"]
    assert len(ai["recommended_formation"]) > 0
    assert len(ai["tactical_rationale"]) > 0
    assert len(ai["xg_timeline"]) == 90


def test_match_scenario_simulator_vulnerabilities_audit():
    """Verifica que el auditor táctico detecta riesgos posicionales reales (ej: pérdidas en salida o espalda de laterales)."""
    sim = MatchScenarioSimulator()
    res = sim.simulate_situational_match(
        home_team="Barcelona",
        away_team="Bayern Munich",
        coach_formation="4-3-3",
        coach_style="Posesión y Bloque Alto",
        opponent_profile={"pressing_intensity": 8.5, "attack_threat": 1.3, "defense_weakness": 1.1, "style": "Presión Alta"}
    )
    alerts = res["own_vulnerabilities_alert"]
    assert len(alerts) > 0
    alert_zones = [a["zona"] for a in alerts]
    assert any("Salida de Balón" in z or "Laterales" in z for z in alert_zones)


def test_academy_scout_prospects_filtering(real_df):
    """Verifica que el ojeador de cantera detecta promesas sub-21 con alto potencial y calcula ahorros."""
    scout = AcademyScout(real_df)
    prospects = scout.get_academy_prospects(min_potential=85, max_age=21)

    assert not prospects.empty
    assert (prospects["potential_ovr"] >= 85).all()
    assert (prospects["age"] <= 21).all()
    assert "growth_delta" in prospects.columns
    assert "projected_savings_m" in prospects.columns
    assert (prospects["projected_savings_m"] >= 0).all()


def test_academy_scout_own_vs_rival(real_df):
    """Verifica el filtrado entre cantera propia (ej: Barcelona / La Masia) y academias externas."""
    scout = AcademyScout(real_df)
    own_barca = scout.get_academy_prospects(team_name="Barcelona", only_own_academy=True)

    assert not own_barca.empty
    assert (own_barca["team_name"] == "Barcelona").all()
    assert any("La Masia" in str(orig) for orig in own_barca["academy_origin"])

    # Todos los canteranos con equipo de referencia tienen marca is_own_academy
    all_with_ref = scout.get_academy_prospects(team_name="Barcelona", only_own_academy=False)
    assert "is_own_academy" in all_with_ref.columns
    assert (all_with_ref[all_with_ref["team_name"] == "Barcelona"]["is_own_academy"]).all()


def test_academy_career_pathway(real_df):
    """Verifica la recomendación de ruta de carrera (ascenso, cesión o fichaje) y desarrollo físico/táctico."""
    scout = AcademyScout(real_df)
    # Lamine Yamal (Barcelona, 17 años, crack mundial)
    pathway_yamal = scout.recommend_career_pathway("Lamine Yamal", target_team="Barcelona")
    assert pathway_yamal["recommended_pathway"] == "Ascenso Inmediato al Primer Equipo"
    assert pathway_yamal["readiness_score_pct"] >= 80
    assert len(pathway_yamal["development_milestones"]) > 0

    # Quenda (Sporting CP, promesa externa vista desde Barcelona)
    pathway_quenda = scout.recommend_career_pathway("Geovany Quenda", target_team="Barcelona")
    assert "Fichaje" in pathway_quenda["recommended_pathway"] or "Observación" in pathway_quenda["recommended_pathway"]


def test_academy_vs_market_comparison(real_df):
    """Verifica el análisis financiero comparativo de canterano vs fichaje senior millonario."""
    scout = AcademyScout(real_df)
    comp = scout.compare_academy_vs_market_signing(
        academy_player_name="Marc Casadó",
        senior_star_name="Rodri"
    )

    assert "academy_prospect" in comp
    assert "senior_star" in comp
    assert "financial_verdict" in comp

    verdict = comp["financial_verdict"]
    assert verdict["net_savings_eur_m"] > 0
    assert verdict["ffp_relief_annual_m"] > 0
