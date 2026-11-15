from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import log_loss


class ModelEvaluator:
    """
    Motor matemático de Evaluación y Validación de Modelos Predictivos en Fútbol.
    Calcula métricas probabilísticas multiclase:
      - Log Loss (entropía cruzada)
      - Brier Score multiclase (error cuadrático medio probabilístico)
      - Ranked Probability Score (RPS) de Epstein, estándar de oro para pronósticos ordenados en fútbol
      - Accuracy / Tasa de acierto del resultado modal
    """

    @staticmethod
    def calculate_rps(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calcula el Ranked Probability Score (RPS) para eventos ordenados (Victoria Local, Empate, Victoria Visitante).
        Fórmula de Epstein (1969) / Constantinou & Fenton (2012):
        RPS = 1 / (K - 1) * sum_{i=1}^{K-1} ( sum_{j=1}^i p_j - sum_{j=1}^i e_j )^2
        donde K=3.
        """
        y_true_arr = np.asarray(y_true, dtype=float)
        y_prob_arr = np.asarray(y_prob, dtype=float)

        # Normalizar probabilidades por fila si fuera necesario
        row_sums = y_prob_arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        y_prob_norm = y_prob_arr / row_sums

        cum_prob = np.cumsum(y_prob_norm, axis=1)
        cum_true = np.cumsum(y_true_arr, axis=1)

        diff = cum_prob[:, :-1] - cum_true[:, :-1]
        rps = np.mean(0.5 * np.sum(diff ** 2, axis=1))
        return float(rps)

    @classmethod
    def calculate_metrics(cls, y_true: Union[np.ndarray, list], y_prob: Union[np.ndarray, list]) -> Dict[str, float]:
        """
        Evalúa el rendimiento de las predicciones 1X2.
        y_true: array de shape (N, 3) con resultados reales codificados (one-hot) o etiquetas de clase.
        y_prob: array de shape (N, 3) con las probabilidades estimadas por el modelo.
        """
        y_true_arr = np.asarray(y_true, dtype=float)
        y_prob_arr = np.asarray(y_prob, dtype=float)

        if y_true_arr.ndim == 1:
            # Convertir etiquetas discretas a one-hot (0, 1, 2)
            n_classes = 3
            n_samples = len(y_true_arr)
            one_hot = np.zeros((n_samples, n_classes))
            for i, val in enumerate(y_true_arr):
                one_hot[i, int(val)] = 1.0
            y_true_arr = one_hot

        # Normalizar sumas de probabilidades para cumplir con el simplex
        row_sums = y_prob_arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        y_prob_norm = y_prob_arr / row_sums

        # Log loss (penalización estricta por descalibración)
        try:
            ll = float(log_loss(y_true_arr, y_prob_norm))
        except Exception:
            clipped = np.clip(y_prob_norm, 1e-15, 1.0 - 1e-15)
            ll = float(-np.mean(np.sum(y_true_arr * np.log(clipped), axis=1)))

        # Brier score multiclase (error cuadrático medio de las probabilidades)
        brier = float(np.mean(np.sum((y_prob_norm - y_true_arr) ** 2, axis=1)))

        # Ranked Probability Score
        rps = cls.calculate_rps(y_true_arr, y_prob_norm)

        # Accuracy modal
        pred_labels = np.argmax(y_prob_norm, axis=1)
        true_labels = np.argmax(y_true_arr, axis=1)
        accuracy = float(np.mean(pred_labels == true_labels))

        return {
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4),
            "rps": round(rps, 4),
            "accuracy": round(accuracy, 4)
        }
