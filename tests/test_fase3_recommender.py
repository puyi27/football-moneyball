import pytest
import pandas as pd
from pytest import approx
from src.models_scouting.recommender import ScoutingRecommender

@pytest.fixture
def mock_metrics_data():
    """Datos simulados de salida del pipeline de métricas."""
    return pd.DataFrame({
        'player_id': [1, 2, 3, 4],
        'player_name': ['Goleador A', 'Goleador B', 'Defensivo C', 'Joven D (Pocos min)'],
        'is_reliable': [True, True, True, False],
        'goals_p90': [0.9, 0.8, 0.0, 1.5],
        'assists_p90': [0.1, 0.2, 0.1, 0.0]
    })

def test_weighted_score(mock_metrics_data):
    """Verifica que el ranking ponderado suma correctamente los pesos."""
    recommender = ScoutingRecommender(mock_metrics_data)
    
    # Pesos: 1 punto por gol_p90, 0.5 puntos por asistencia_p90
    weights = {'goals_p90': 1.0, 'assists_p90': 0.5}
    df_scored = recommender.calculate_weighted_score(weights)
    
    # Goleador A: (0.9 * 1.0) + (0.1 * 0.5) = 0.95
    # Goleador B: (0.8 * 1.0) + (0.2 * 0.5) = 0.90
    assert df_scored.iloc[0]['player_id'] == 1, "Goleador A debe ser el primero"
    assert df_scored.iloc[0]['custom_score'] == approx(0.95), "Error en el cálculo aritmético del peso"

def test_find_similar_players(mock_metrics_data):
    """Verifica que KNN encuentra al perfil más cercano."""
    recommender = ScoutingRecommender(mock_metrics_data)
    
    # Buscar similar al Goleador A (id=1). El más cercano lógicamente es Goleador B (id=2)
    similar = recommender.find_similar_players(player_id=1, features=['goals_p90', 'assists_p90'], k=1)
    
    assert len(similar) == 1, "Debe retornar exactamente k jugadores"
    assert similar.iloc[0]['player_id'] == 2, "KNN falló al encontrar el perfil más similar"

def test_unreliable_player_exclusion(mock_metrics_data):
    """Verifica que los jugadores no fiables son excluidos del motor."""
    recommender = ScoutingRecommender(mock_metrics_data)
    
    # El jugador 4 no es fiable, debería lanzar ValueError si intentamos buscarlo
    with pytest.raises(ValueError, match="no tiene minutos suficientes"):
        recommender.find_similar_players(player_id=4, features=['goals_p90'], k=1)