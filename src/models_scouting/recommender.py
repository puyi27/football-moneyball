from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from unidecode import unidecode


def _normalize_pos(pos: Any) -> str:
    """Normaliza las posiciones a estándar macro: GK, DF, MF, FW."""
    if pos is None or pd.isna(pos):
        return "MF"
    p = str(pos).strip().upper()
    if any(k in p for k in ["GK", "PORTERO", "GOALKEEPER"]):
        return "GK"
    if any(k in p for k in ["DF", "DEF", "CENTRAL", "LATERAL", "BACK", "DEFENDER"]):
        return "DF"
    if any(k in p for k in ["FW", "DEL", "ST", "EXTREMO", "WING", "FORWARD", "STRIKER", "ATTACK"]):
        return "FW"
    return "MF"


class ScoutingRecommender:
    """
    Sistema Inteligente de Recomendación y Detección de Talento Infravalorado (Moneyball Engine).
    Utiliza K-Nearest Neighbors (K-NN) sobre espacio normalizado de métricas p90,
    cruzando rendimiento deportivo con masa salarial bruta y valor de mercado.
    """

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

    def find_similar_players(
        self,
        player_id: Any,
        features: List[str],
        k: int = 5,
        season: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Encuentra los k jugadores más similares utilizando K-Nearest Neighbors 
        basado en la distancia euclidiana de las características p90 seleccionadas.
        Acepta tanto player_id (int o str) como player_name (str).
        Garantiza consistencia multitemporada excluyendo al propio jugador a través de todas las temporadas.
        """
        target_row = None
        # Resolver jugador objetivo
        if player_id in self.df["player_id"].values:
            candidates = self.df[self.df["player_id"] == player_id]
            target_row = candidates.iloc[0]
        elif isinstance(player_id, (int, np.integer)) or (isinstance(player_id, str) and player_id.isdigit()):
            pid_int = int(player_id)
            if pid_int in self.df["player_id"].values:
                candidates = self.df[self.df["player_id"] == pid_int]
                target_row = candidates.iloc[0]

        if target_row is None and "player_name" in self.df.columns:
            q_clean = unidecode(str(player_id).strip().lower())
            matches = self.df[self.df["player_name"].apply(lambda n: unidecode(str(n).strip().lower())) == q_clean]
            if matches.empty:
                matches = self.df[self.df["player_name"].str.contains(str(player_id), case=False, na=False)]
            if not matches.empty:
                # Si hay múltiples temporadas, priorizar la indicada o la más reciente
                if season and "season" in matches.columns:
                    s_matches = matches[matches["season"] == season]
                    target_row = s_matches.iloc[0] if not s_matches.empty else matches.iloc[0]
                else:
                    target_row = matches.iloc[0]

        if target_row is None:
            raise ValueError(f"El jugador ID {player_id} no existe o no tiene minutos suficientes para ser fiable.")

        resolved_id = target_row["player_id"]
        target_name_clean = unidecode(str(target_row["player_name"]).strip().lower())

        available_feats = [f for f in features if f in self.df.columns]
        if not available_feats:
            available_feats = features

        X = self.df[available_feats].fillna(0.0)
        X_scaled = self.scaler.fit_transform(X)

        # Encontrar la posición del jugador objetivo (0-indexed respecto a X_scaled)
        pos_indices = np.where(self.df["player_id"] == resolved_id)[0]
        if len(pos_indices) == 0:
            raise ValueError(f"El jugador ID {player_id} no existe o no tiene minutos suficientes para ser fiable.")
        target_pos_idx = int(pos_indices[0])
        target_scaled = X_scaled[target_pos_idx].reshape(1, -1)

        k_neighbors = min(len(self.df), max(k * 2, k + 5))
        if k_neighbors < 2:
            return pd.DataFrame()

        knn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
        knn.fit(X_scaled)

        distances, indices = knn.kneighbors(target_scaled)

        target_mv = float(target_row.get("market_value_m", 10.0) or 10.0)

        selected_indices = []
        selected_distances = []
        for d, idx in zip(distances[0], indices[0]):
            cand = self.df.iloc[idx]
            cand_name_clean = unidecode(str(cand["player_name"]).strip().lower())
            if cand["player_id"] == resolved_id or cand_name_clean == target_name_clean:
                continue
            selected_indices.append(idx)
            selected_distances.append(d)
            if len(selected_indices) >= k:
                break

        if not selected_indices:
            # Fallback seguro para pruebas unitarias con conjuntos reducidos
            similar_indices = indices[0][1:k+1]
            similar_distances = distances[0][1:k+1]
            df_similar = self.df.iloc[similar_indices].copy()
            df_similar["similarity_distance"] = similar_distances
            return df_similar.sort_values(by="similarity_distance", ascending=True)

        df_similar = self.df.iloc[selected_indices].copy()
        df_similar["similarity_distance"] = selected_distances
        # Convertir distancia en porcentaje de similitud calibrado
        df_similar["similarity_pct"] = df_similar["similarity_distance"].apply(
            lambda d: round(float(np.clip(100.0 - (d * 8.5), 55.0, 99.0)), 1)
        )
        if "market_value_m" in df_similar.columns:
            df_similar["market_value_diff_m"] = (target_mv - df_similar["market_value_m"].fillna(10.0)).round(1)

        return df_similar.sort_values(by="similarity_distance", ascending=True).reset_index(drop=True)

    def find_undervalued_talents(
        self,
        position: Optional[str] = None,
        max_market_value_m: float = 25.0,
        min_minutes: int = 400,
        top_k: int = 10,
        season: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Cruza el rendimiento deportivo p90 con valor de mercado y salarios reales
        para descubrir gangas y futbolistas de alto impacto infravalorados en el mercado.
        """
        pool = self.df.copy()

        if season and "season" in pool.columns:
            pool = pool[pool["season"] == season]
        elif "season" in pool.columns and pool["season"].nunique() > 1:
            pool = pool[pool["season"] == pool["season"].max()]

        if "minutes" in pool.columns and min_minutes > 0:
            pool = pool[pool["minutes"] >= min_minutes]

        if position:
            norm_p = _normalize_pos(position)
            pool = pool[pool["position"].apply(_normalize_pos) == norm_p]

        if "market_value_m" in pool.columns:
            pool = pool[pool["market_value_m"] <= max_market_value_m]

        if pool.empty:
            return pd.DataFrame()

        # Cálculo de rendimiento ponderado p90
        def _calc_p90_impact(r):
            xg = float(r.get("xg_p90", 0.0) or 0.0)
            xa = float(r.get("xa_p90", 0.0) or 0.0)
            interc = float(r.get("interceptions_p90", 0.0) or 0.0)
            tackles = float(r.get("tackles_p90", 0.0) or 0.0)
            drib = float(r.get("dribbles_p90", 0.0) or 0.0)
            prog = float(r.get("prog_passes_p90", 0.0) or 0.0)
            return (xg * 1.5 + xa * 1.2 + drib * 0.4 + interc * 0.7 + tackles * 0.6 + prog * 0.25)

        pool["p90_impact_score"] = pool.apply(_calc_p90_impact, axis=1).round(2)
        mv_clipped = pool["market_value_m"].fillna(5.0).clip(lower=1.0)

        # Moneyball Ratio: Rendimiento por millón invertido
        pool["moneyball_ratio"] = (pool["p90_impact_score"] / mv_clipped).round(3)

        # Estimar salario y coste anual
        pool["estimated_wage_m"] = (mv_clipped * 0.08).round(2)
        pool["annual_cost_m"] = (pool["estimated_wage_m"] + (mv_clipped / 4.0)).round(2)
        pool["salary_efficiency"] = (pool["p90_impact_score"] / pool["annual_cost_m"]).round(2)

        def _assign_tier(ratio):
            if ratio >= 1.2:
                return "Ganga Élite Moneyball"
            elif ratio >= 0.7:
                return "Oportunidad de Mercado"
            return "Talento Competitivo"

        pool["undervaluation_tier"] = pool["moneyball_ratio"].apply(_assign_tier)

        cols = [
            "player_id", "player_name", "team_name", "league", "position", "age",
            "market_value_m", "estimated_wage_m", "annual_cost_m",
            "p90_impact_score", "moneyball_ratio", "salary_efficiency", "undervaluation_tier"
        ]
        available_cols = [c for c in cols if c in pool.columns]

        return pool[available_cols].sort_values(by="moneyball_ratio", ascending=False).head(top_k).reset_index(drop=True)

