import pytest
import pandas as pd
from src.models_analytics.optimizer import PoissonOptimizer

def test_poisson_optimizer_convergence():
    """Prueba que el optimizador MLE logra converger con datos simulados."""
    # Simulación de una mini-liga
    data = {
        'home_team': ['Equipo A', 'Equipo B', 'Equipo C', 'Equipo A', 'Equipo B', 'Equipo C'],
        'away_team': ['Equipo B', 'Equipo C', 'Equipo A', 'Equipo C', 'Equipo A', 'Equipo B'],
        'home_goals': [2, 1, 0, 3, 1, 0],
        'away_goals': [0, 1, 2, 0, 2, 1]
    }
    df_matches = pd.DataFrame(data)
    
    opt = PoissonOptimizer()
    params = opt.fit(df_matches)
    
    # 1. Comprobar que todos los parámetros existen
    assert 'attack' in params
    assert 'defense' in params
    assert 'home_adv' in params
    assert 'rho' in params
    
    # 2. Comprobar restricción de identificabilidad (la suma de ataques debe ser ~0)
    sum_attacks = sum(params['attack'].values())
    assert sum_attacks == pytest.approx(0.0, abs=1e-3), "La suma de ataques no es 0"
    
    # 3. La ventaja de local no debe ser negativa según los bounds
    assert params['home_adv'] >= 0.0, "La ventaja de local debe ser >= 0"
    
    # 4. Equipo A ganó todos sus partidos con goleadas, debe tener el mejor ataque
    assert params['attack']['Equipo A'] > params['attack']['Equipo C']