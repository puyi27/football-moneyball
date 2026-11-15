from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class PlayerProfiler:
    """
    Motor de Perfilado 360° del Futbolista.
    Calcula la carta de atributos FIFA/EA FC (1-99 y OVR),
    mapas de zonas de influencia sobre el terreno de juego,
    búsqueda de 'Jugadores Clones / Gemelos' (K-NN con % similitud)
    y el Moneyball Value Ratio (rendimiento por millón de euros).
    """

    def __init__(self, league_reference_df: pd.DataFrame):
        self.df = league_reference_df.copy()

    @staticmethod
    def _scale_to_stat(val: float, min_val: float, max_val: float, floor: int = 50, ceiling: int = 99) -> int:
        """Escala una métrica deportiva a una puntuación FIFA en la escala de 50 a 99."""
        if pd.isna(val) or val is None:
            return floor
        clipped = np.clip(val, min_val, max_val)
        norm = (clipped - min_val) / max(max_val - min_val, 1e-6)
        score = int(round(floor + norm * (ceiling - floor)))
        return int(np.clip(score, floor, ceiling))

    def calculate_fifa_card(self, player_series: pd.Series) -> Dict[str, Any]:
        """
        Calcula las 6 macro-habilidades FIFA (PAC, SHO, PAS, DRI, DEF, PHY)
        y la valoración global (OVR) ponderada según la demarcación.
        """
        xg = float(player_series.get("xg_p90", 0.0) or 0.0)
        xa = float(player_series.get("xa_p90", 0.0) or 0.0)
        goals = float(player_series.get("goals_p90", 0.0) or 0.0)
        assists = float(player_series.get("assists_p90", 0.0) or 0.0)
        interc = float(player_series.get("interceptions_p90", 0.0) or 0.0)
        tackles = float(player_series.get("tackles_p90", 0.0) or 0.0)
        dribbles = float(player_series.get("dribbles_p90", 1.5) or 1.5)
        prog_passes = float(player_series.get("prog_passes_p90", 4.0) or 4.0)
        pos = str(player_series.get("position", "MF")).upper()
        mins = float(player_series.get("minutes", 1500) or 1500)

        # 1. SHO (Finalización)
        sho_metric = (xg * 0.6) + (goals * 0.4)
        sho = self._scale_to_stat(sho_metric, 0.05, 0.85, floor=48, ceiling=97)

        # 2. PAS (Pase y Creación)
        pas_metric = (xa * 0.5) + (assists * 0.25) + (prog_passes * 0.05)
        pas = self._scale_to_stat(pas_metric, 0.08, 0.85, floor=52, ceiling=96)

        # 3. DRI (Regate y Desequilibrio)
        dri_metric = (dribbles * 0.7) + (xa * 1.5)
        dri = self._scale_to_stat(dri_metric, 0.5, 4.8, floor=50, ceiling=98)

        # 4. DEF (Defensa y Recuperación)
        def_metric = (interc * 0.55) + (tackles * 0.45)
        deff = self._scale_to_stat(def_metric, 0.4, 2.8, floor=45, ceiling=95)

        # 5. PAC (Ritmo y Progresión)
        pac_metric = (dribbles * 0.5) + (prog_passes * 0.3)
        pac = self._scale_to_stat(pac_metric, 1.0, 5.5, floor=55, ceiling=98)

        # 6. PHY (Físico y Duelos)
        phy_metric = (tackles * 0.4) + (interc * 0.3) + (mins / 2500.0 * 0.8)
        phy = self._scale_to_stat(phy_metric, 0.8, 2.5, floor=54, ceiling=93)

        # 7. OVR (Overall Ponderado por posición)
        if "FW" in pos:
            ovr = int(round(sho * 0.35 + dri * 0.25 + pac * 0.20 + pas * 0.15 + phy * 0.05))
        elif "DF" in pos:
            ovr = int(round(deff * 0.45 + phy * 0.25 + pac * 0.15 + pas * 0.15))
        elif "GK" in pos:
            ovr = int(round(deff * 0.50 + phy * 0.30 + pac * 0.20))
        else:  # Centrocampista (MF)
            ovr = int(round(pas * 0.30 + dri * 0.22 + deff * 0.20 + phy * 0.15 + sho * 0.13))

        ovr = int(np.clip(ovr, 65, 98))

        return {
            "OVR": ovr,
            "PAC": pac,
            "SHO": sho,
            "PAS": pas,
            "DRI": dri,
            "DEF": deff,
            "PHY": phy,
            "player_name": player_series.get("player_name", "Desconocido"),
            "position": pos,
            "team_name": player_series.get("team_name", "Sin equipo"),
            "league": player_series.get("league", "Liga"),
            "age": int(player_series.get("age", 25)),
            "market_value_m": float(player_series.get("market_value_m", 10.0)),
        }

    def get_pitch_heatmap_data(self, player_series: pd.Series, n_samples: int = 180) -> pd.DataFrame:
        """
        Genera coordenadas espaciales (X: 0-105m, Y: 0-68m) representando la zona de calor
        de actividad preferente del futbolista según su demarcación y estilo de juego.
        """
        pos = str(player_series.get("position", "MF")).upper()
        pname = str(player_series.get("player_name", ""))
        dribbles = float(player_series.get("dribbles_p90", 2.0))

        np.random.seed(abs(hash(pname)) % 10000)

        # Coordenadas centroide y dispersión según demarcación
        if "DF" in pos:
            # Centrales / Defensas: tercio defensivo
            x_center = np.random.uniform(22, 38)
            y_center = 34.0
            x_std, y_std = 9.0, 15.0
        elif "FW" in pos:
            # Delanteros y Extremos
            if "Vinícius" in pname or "Nico" in pname or "Leão" in pname or "Barcola" in pname:
                # Extremos zurdos (carril izquierdo alto)
                x_center, y_center = 78.0, 52.0
                x_std, y_std = 12.0, 8.0
            elif "Yamal" in pname or "Saka" in pname or "Kubo" in pname or "Bakayoko" in pname:
                # Extremos derechos (carril derecho alto)
                x_center, y_center = 80.0, 16.0
                x_std, y_std = 12.0, 8.0
            else:
                # Delanteros centro (Mbappé, Haaland, Kane, Lewandowski)
                x_center, y_center = 84.0, 34.0
                x_std, y_std = 11.0, 12.0
        else:  # Centrocampistas
            if "Rodri" in pname or "Zubimendi" in pname or "Schouten" in pname:
                # Pivotes defensivos (círculo central y cobertura por detrás del balón)
                x_center, y_center = 50.0, 34.0
                x_std, y_std = 14.0, 16.0
            else:
                # Interiores / Mediapuntas (Pedri, Bellingham, Wirtz, Musiala, Isco)
                x_center, y_center = 65.0, 34.0
                x_std, y_std = 15.0, 18.0

        x_coords = np.clip(np.random.normal(x_center, x_std, n_samples), 5.0, 100.0)
        y_coords = np.clip(np.random.normal(y_center, y_std, n_samples), 4.0, 64.0)

        return pd.DataFrame({"x": x_coords, "y": y_coords})

    def find_player_twins(
        self,
        player_id: int,
        top_k: int = 5,
        exclude_same_team: bool = True
    ) -> pd.DataFrame:
        """
        Encuentra los futbolistas más similares en toda Europa ('Jugadores Clones / Gemelos')
        usando K-Nearest Neighbors y calcula el porcentaje de similitud real y el diferencial de precio.
        """
        if player_id not in self.df["player_id"].values:
            raise ValueError(f"Jugador ID {player_id} no encontrado en la base de datos.")

        target = self.df[self.df["player_id"] == player_id].iloc[0]
        pool = self.df.copy()

        if exclude_same_team and "team_name" in target:
            pool = pool[pool["team_name"] != target["team_name"]]

        # Filtrar jugadores de la misma posición macro
        target_pos = target["position"]
        pool = pool[pool["position"] == target_pos]

        features = ["xg_p90", "xa_p90", "interceptions_p90", "tackles_p90", "dribbles_p90", "prog_passes_p90"]
        available_feats = [f for f in features if f in pool.columns]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pool[available_feats].fillna(0))

        # Vector del jugador objetivo
        target_vector = scaler.transform(target[available_feats].to_frame().T.fillna(0))

        k_neighbors = min(top_k + 1, len(pool))
        knn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
        knn.fit(X_scaled)

        distances, indices = knn.kneighbors(target_vector)
        dist_list = distances[0]
        idx_list = indices[0]

        results = []
        for d, idx in zip(dist_list, idx_list):
            cand = pool.iloc[idx]
            if cand["player_id"] == player_id:
                continue

            # Convertir distancia euclidiana en porcentaje de similitud (99% = idéntico)
            sim_pct = round(float(np.clip(100.0 - (d * 14.5), 60.0, 99.2)), 1)
            target_mv = float(target.get("market_value_m", 10.0))
            cand_mv = float(cand.get("market_value_m", 10.0))
            saving_m = round(target_mv - cand_mv, 1)

            results.append({
                "player_id": int(cand["player_id"]),
                "player_name": cand["player_name"],
                "team_name": cand["team_name"],
                "league": cand["league"],
                "position": cand["position"],
                "age": int(cand["age"]),
                "similarity_pct": sim_pct,
                "market_value_m": cand_mv,
                "savings_vs_target_m": saving_m,
                "xg_p90": round(float(cand.get("xg_p90", 0.0)), 2),
                "xa_p90": round(float(cand.get("xa_p90", 0.0)), 2),
                "interceptions_p90": round(float(cand.get("interceptions_p90", 0.0)), 2),
            })

            if len(results) >= top_k:
                break

        return pd.DataFrame(results).sort_values(by="similarity_pct", ascending=False)

    def calculate_moneyball_bargains(self, min_minutes: int = 600, top_n: int = 10) -> pd.DataFrame:
        """
        Calcula el 'Moneyball Index': ratio de rendimiento ofensivo + defensivo por millón de euros.
        Identifica gangas o jugadores infravalorados ideales para clubes con presupuesto ajustado.
        """
        df_eval = self.df[self.df["minutes"] >= min_minutes].copy()

        # Índice de rendimiento = (xG + xA) * 1.5 + Intercepciones * 0.8
        df_eval["performance_index"] = (
            (df_eval["xg_p90"].fillna(0) + df_eval["xa_p90"].fillna(0)) * 1.5 +
            df_eval["interceptions_p90"].fillna(0) * 0.8
        )

        # Moneyball Ratio = Rendimiento / Valor de Mercado (M€)
        df_eval["moneyball_ratio"] = df_eval["performance_index"] / df_eval["market_value_m"].clip(lower=1.0)
        df_eval["moneyball_ratio"] = df_eval["moneyball_ratio"].round(3)

        cols = ["player_name", "team_name", "league", "position", "age", "market_value_m", "performance_index", "moneyball_ratio"]
        return df_eval[cols].sort_values(by="moneyball_ratio", ascending=False).head(top_n)
