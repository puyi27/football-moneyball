from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class ScoutingRecommender:
    def __init__(self, df_metrics: Optional[pd.DataFrame] = None):
        """
        Inicializa el recomendador filtrando únicamente a los jugadores fiables
        (aquellos que superan el umbral de minutos establecido en el pipeline de métricas).
        Si df_metrics es None, carga la base de datos global de jugadores.
        """
        if df_metrics is None or df_metrics.empty:
            from src.data.global_player_database import load_global_players
            df_metrics = load_global_players()

        if "is_reliable" in df_metrics.columns:
            reliable = df_metrics[df_metrics["is_reliable"] == True]
            self.df = reliable.copy().reset_index(drop=True) if not reliable.empty else df_metrics.copy().reset_index(drop=True)
        else:
            self.df = df_metrics.copy().reset_index(drop=True)

        self.scaler = StandardScaler()

    def calculate_weighted_score(self, weights: Dict[str, float]) -> pd.DataFrame:
        """
        Genera un ranking basado en pesos personalizados por el scout.
        Ejemplo de weights: {'goals_p90': 1.0, 'assists_p90': 0.5}
        """
        self.df["custom_score"] = 0.0
        for metric, weight in weights.items():
            if metric in self.df.columns:
                self.df["custom_score"] += self.df[metric].fillna(0.0) * float(weight)

        return self.df.sort_values(by="custom_score", ascending=False)

    def find_similar_players(self, player_id: Any, features: List[str], k: int = 5) -> pd.DataFrame:
        """
        Encuentra los k jugadores más similares utilizando K-Nearest Neighbors 
        basado en la distancia euclidiana de las características seleccionadas.
        Acepta tanto player_id (int) como player_name (str).
        """
        if player_id not in self.df["player_id"].values:
            if "player_name" in self.df.columns and player_id in self.df["player_name"].values:
                target_row = self.df[self.df["player_name"] == player_id].iloc[0]
                player_id = target_row["player_id"]
            else:
                raise ValueError(f"El jugador ID {player_id} no existe o no tiene minutos suficientes para ser fiable.")

        available_feats = [f for f in features if f in self.df.columns]
        if not available_feats:
            available_feats = features

        X = self.df[available_feats].fillna(0.0)
        X_scaled = self.scaler.fit_transform(X)

        k_val = min(k, len(self.df) - 1)
        if k_val < 1:
            return pd.DataFrame()

        knn = NearestNeighbors(n_neighbors=k_val + 1, metric="euclidean")
        knn.fit(X_scaled)

        # Encontrar la posición del jugador objetivo (0-indexed respecto a X_scaled)
        pos_indices = np.where(self.df["player_id"] == player_id)[0]
        if len(pos_indices) == 0:
            raise ValueError(f"El jugador ID {player_id} no existe o no tiene minutos suficientes para ser fiable.")
        target_pos_idx = int(pos_indices[0])
        target_scaled = X_scaled[target_pos_idx].reshape(1, -1)

        distances, indices = knn.kneighbors(target_scaled)

        similar_indices = indices[0][1:]
        similar_distances = distances[0][1:]

        df_similar = self.df.iloc[similar_indices].copy()
        df_similar["similarity_distance"] = similar_distances

        return df_similar.sort_values(by="similarity_distance", ascending=True)
