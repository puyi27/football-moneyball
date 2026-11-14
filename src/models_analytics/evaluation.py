import numpy as np
from sklearn.metrics import log_loss

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        """
        Evalúa el rendimiento de las predicciones 1X2.
        y_true: array de shape (N, 3) con resultados reales codificados (one-hot).
        y_prob: array de shape (N, 3) con las probabilidades estimadas por el modelo.
        """
        # Log loss (Penaliza fuertemente la confianza extrema en predicciones erróneas)
        ll = log_loss(y_true, y_prob)
        
        # Brier score multiclase (Error cuadrático medio de las probabilidades)
        brier = np.mean(np.sum((y_prob - y_true)**2, axis=1))
        
        return {
            "log_loss": float(ll), 
            "brier_score": float(brier)
        }