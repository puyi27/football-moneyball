import pytest
import pandas as pd
from src.models_scouting.squad_diagnostics import SquadDiagnostics

@pytest.fixture
def mock_squad_data():
    """Datos simulados para análisis de plantilla."""
    return pd.DataFrame({
        'player_id': [1, 2, 3, 4],
        'player_name': ['Avanzado A', 'Veterano B', 'Rival C', 'Rival D'],
        'team_name': ['Real Betis', 'Real Betis', 'Sevilla FC', 'Sevilla FC'],
        'position': ['Forward', 'Defender', 'Forward', 'Defender'],
        'age': [24, 33, 26, 22],
        'minutes': [800, 900, 700, 600],
        'goals_p90': [0.8, 0.0, 0.5, 0.1]
    })

def test_analyze_team_deficits(mock_squad_data):
    """Verifica el cálculo de déficits por posición comparado con la liga."""
    diag = SquadDiagnostics(mock_squad_data)
    df_deficits = diag.analyze_team_deficits(team_name='Real Betis', metric='goals_p90')

    # Promedio liga Forwards: (0.8 + 0.5) / 2 = 0.65. Betis Forwards: 0.8 -> Déficit = +0.15 (Sin necesidad)
    # Promedio liga Defenders: (0.0 + 0.1) / 2 = 0.05. Betis Defenders: 0.0 -> Déficit = -0.05 (Necesidad)
    defenders_row = df_deficits[df_deficits['position'] == 'Defender'].iloc[0]
    assert defenders_row['needs_reinforcement'] == True, "La posición Defender debería necesitar refuerzo"
    assert defenders_row['deficit'] == pytest.approx(-0.05), "Error en el cálculo del déficit"

def test_detect_squad_risk_alerts(mock_squad_data):
    """Verifica la detección de alertas por veteranía y concentración de minutos."""
    diag = SquadDiagnostics(mock_squad_data)
    alerts = diag.detect_squad_risk_alerts(team_name='Real Betis', max_age=30, min_minutes_share=0.5)

    assert len(alerts) == 1, "Debe detectar exactamente una alerta en el Real Betis"
    assert alerts.iloc[0]['player_name'] == 'Veterano B', "La alerta debe pertenecer al Veterano B"