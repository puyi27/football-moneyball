import pytest
import numpy as np
from src.models_analytics.elo_model import EloSystem
from src.models_analytics.poisson_model import DixonColesModel

def test_elo_update_logic():
    elo = EloSystem(k_factor=20, home_advantage=0) # Anulamos localía para test puro
    
    # Victoria del local
    elo.update_ratings("Equipo A", "Equipo B", 2, 0)
    
    assert elo.get_rating("Equipo A") > 1500.0
    assert elo.get_rating("Equipo B") < 1500.0

def test_poisson_coherence():
    dc = DixonColesModel(max_goals=10)
    matrix = dc.calculate_match_matrix(lambda_=1.5, mu=1.2, rho=0.0)
    
    # La probabilidad total del espacio muestral debe ser ~1.0
    assert matrix.sum() == pytest.approx(1.0, abs=1e-3)
    
    # Probabilidades 1X2 deben sumar 1.0
    h, d, a = dc.get_1x2_probabilities(matrix)
    assert (h + d + a) == pytest.approx(1.0, abs=1e-3)

def test_dixon_coles_draw_inflation():
    dc = DixonColesModel(max_goals=5)
    
    # Sin ajuste (independiente)
    matrix_indep = dc.calculate_match_matrix(1.2, 1.2, rho=0.0)
    draw_prob_indep = np.trace(matrix_indep)
    
    # Con ajuste Dixon-Coles (rho negativo típico incrementa empates de baja anotación)
    matrix_dc = dc.calculate_match_matrix(1.2, 1.2, rho=-0.15)
    draw_prob_dc = np.trace(matrix_dc)
    
    assert draw_prob_dc > draw_prob_indep