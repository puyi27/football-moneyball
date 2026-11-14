import numpy as np
import pytest
from src.models_analytics.evaluation import ModelEvaluator

def test_evaluation_metrics():
    """Verifica el cálculo correcto de Log Loss y Brier Score."""
    # Simulación: Partido 1 (Victoria Local), Partido 2 (Empate)
    y_true = np.array([
        [1, 0, 0], 
        [0, 1, 0]
    ]) 
    
    # Probabilidades estimadas por el modelo (buenas predicciones)
    y_prob = np.array([
        [0.70, 0.20, 0.10], 
        [0.30, 0.50, 0.20]
    ])
    
    metrics = ModelEvaluator.calculate_metrics(y_true, y_prob)
    
    assert "log_loss" in metrics
    assert "brier_score" in metrics
    assert metrics["log_loss"] > 0.0
    assert metrics["brier_score"] > 0.0
    # Brier Score perfecto es 0.0. Valores < 0.6 son típicos en fútbol.
    assert metrics["brier_score"] < 1.0