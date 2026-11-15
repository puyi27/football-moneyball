from typing import Dict, List, Any, Tuple, Optional, Union
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

    def __init__(self, league_reference_df: Optional[pd.DataFrame] = None):
        if league_reference_df is not None and not league_reference_df.empty:
            self.df = league_reference_df.copy()
        else:
            from src.data.global_player_database import load_global_players
            self.df = load_global_players()

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
        Incluye atributos especializados de portero (DIV, HAN, KIC, REF, SPE, POS).
        """
        pos = str(player_series.get("position", "MF")).upper()
        age_val = player_series.get("age", 25)
        age = int(age_val) if pd.notna(age_val) and age_val is not None else 25
        mv_val = player_series.get("market_value_m", 10.0)
        mv = float(mv_val) if pd.notna(mv_val) and mv_val is not None else 10.0
        mins = float(player_series.get("minutes", 1500) or 1500)
        prog_passes = float(player_series.get("prog_passes_p90", 4.0) or 4.0)
        interc = float(player_series.get("interceptions_p90", 0.0) or 0.0)
        tackles = float(player_series.get("tackles_p90", 0.0) or 0.0)
        dribbles = float(player_series.get("dribbles_p90", 1.5) or 1.5)

        if "GK" in pos:
            # Calibración deportiva para guardametas (DIV, HAN, KIC, REF, SPE, POS)
            pot_val = player_series.get("potential_ovr", 82)
            pot = int(pot_val) if pd.notna(pot_val) and pot_val is not None else 82

            is_starter = (mins >= 1200 or mv >= 15.0)
            if age <= 23:
                base_ovr = int(round(pot - max(0, 24 - age) * 0.7))
            else:
                base_ovr = pot if is_starter else max(pot - 2, 75)
            ovr = int(np.clip(base_ovr, 72, 94))

            # Atributos específicos FIFA de porteros
            div = int(np.clip(round(ovr + (1 if mv >= 20.0 else -1)), 68, 93))
            han = int(np.clip(round(ovr - (1 if age < 24 else 0)), 68, 92))
            kic = int(np.clip(round(64 + prog_passes * 12.0), 62, 90))
            ref = int(np.clip(round(ovr + (2 if age < 30 else 0)), 70, 94))
            spe = int(np.clip(round(48 + (interc * 35.0)), 45, 78))
            pos_stat = int(np.clip(round(ovr + (2 if age >= 28 else -1)), 68, 93))

            # Mapeo estándar a PAC, SHO, PAS, DRI, DEF, PHY
            return {
                "OVR": ovr,
                "PAC": spe,
                "SHO": han,
                "PAS": kic,
                "DRI": ref,
                "DEF": div,
                "PHY": pos_stat,
                "DIV": div,
                "HAN": han,
                "KIC": kic,
                "REF": ref,
                "SPE": spe,
                "POS": pos_stat,
                "is_goalkeeper": True,
                "player_name": player_series.get("player_name", "Desconocido"),
                "position": pos,
                "team_name": player_series.get("team_name", "Sin equipo"),
                "league": player_series.get("league", "Liga"),
                "age": age,
                "market_value_m": mv,
            }

        # Demarcaciones de campo (FW, MF, DF)
        xg = float(player_series.get("xg_p90", 0.0) or 0.0)
        xa = float(player_series.get("xa_p90", 0.0) or 0.0)
        goals = float(player_series.get("goals_p90", 0.0) or 0.0)
        assists = float(player_series.get("assists_p90", 0.0) or 0.0)

        # 1. SHO (Finalización)
        sho_metric = (xg * 0.6) + (goals * 0.4)
        sho = self._scale_to_stat(sho_metric, 0.05, 0.70, floor=58, ceiling=97)

        # 2. PAS (Pase y Creación)
        pas_metric = (xa * 0.5) + (assists * 0.25) + (prog_passes * 0.05)
        pas = self._scale_to_stat(pas_metric, 0.08, 0.65, floor=58, ceiling=96)

        # 3. DRI (Regate y Desequilibrio)
        dri_metric = (dribbles * 0.7) + (xa * 1.5)
        dri = self._scale_to_stat(dri_metric, 0.5, 3.8, floor=58, ceiling=98)

        # 4. DEF (Defensa y Recuperación)
        def_metric = (interc * 0.55) + (tackles * 0.45)
        deff = self._scale_to_stat(def_metric, 0.3, 2.6, floor=50, ceiling=96)

        # 5. PAC (Ritmo y Progresión)
        pac_metric = (dribbles * 0.6) + (prog_passes * 0.25) + (xg * 1.0)
        pac = self._scale_to_stat(pac_metric, 0.8, 4.0, floor=60, ceiling=98)

        # 6. PHY (Físico y Duelos)
        phy_metric = (tackles * 0.4) + (interc * 0.3) + (mins / 2500.0 * 0.8)
        phy = self._scale_to_stat(phy_metric, 0.7, 2.4, floor=56, ceiling=94)

        # 7. OVR (Overall Ponderado por posición)
        if "FW" in pos:
            ovr = int(round(sho * 0.35 + dri * 0.25 + pac * 0.20 + pas * 0.15 + phy * 0.05))
        elif "DF" in pos:
            ovr = int(round(deff * 0.45 + phy * 0.25 + pac * 0.15 + pas * 0.15))
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
            "is_goalkeeper": False,
            "player_name": player_series.get("player_name", "Desconocido"),
            "position": pos,
            "team_name": player_series.get("team_name", "Sin equipo"),
            "league": player_series.get("league", "Liga"),
            "age": age,
            "market_value_m": mv,
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
        if "GK" in pos:
            # Portero: área de meta propia delimitada estrictamente en el área de penalti (0-16.5m)
            x_center = 10.0
            y_center = 34.0
            x_std, y_std = 3.5, 7.0
            x_coords = np.clip(np.random.normal(x_center, x_std, n_samples), 3.0, 24.0)
            y_coords = np.clip(np.random.normal(y_center, y_std, n_samples), 16.0, 52.0)
            return pd.DataFrame({"x": x_coords, "y": y_coords})
        elif "DF" in pos:
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

    def generate_pitch_heatmap(self, player_series: pd.Series, n_samples: int = 180) -> pd.DataFrame:
        """Alias de get_pitch_heatmap_data para compatibilidad con el cuadro de mando."""
        return self.get_pitch_heatmap_data(player_series, n_samples=n_samples)

    def find_player_twins(
        self,
        player_id: Any = None,
        player_name: Optional[str] = None,
        top_k: int = 5,
        n_neighbors: Optional[int] = None,
        exclude_same_team: bool = True
    ) -> pd.DataFrame:
        """
        Encuentra los futbolistas más similares en toda Europa ('Jugadores Clones / Gemelos')
        usando K-Nearest Neighbors y calcula el porcentaje de similitud real y el diferencial de precio.
        Permite invocar tanto por player_id (int) como por player_name (str) o n_neighbors.
        """
        k_count = n_neighbors if n_neighbors is not None else top_k

        # Resolver el jugador objetivo según id o nombre
        if player_name is not None:
            target_mask = self.df["player_name"].str.lower() == str(player_name).lower()
            if not target_mask.any():
                target_mask = self.df["player_name"].str.contains(str(player_name), case=False, na=False)
            if not target_mask.any():
                return pd.DataFrame()
            target = self.df[target_mask].iloc[0]
            resolved_id = target["player_id"]
        elif isinstance(player_id, str) and not player_id.isdigit():
            target_mask = self.df["player_name"].str.lower() == player_id.lower()
            if not target_mask.any():
                target_mask = self.df["player_name"].str.contains(player_id, case=False, na=False)
            if not target_mask.any():
                return pd.DataFrame()
            target = self.df[target_mask].iloc[0]
            resolved_id = target["player_id"]
        else:
            resolved_id = int(player_id) if player_id is not None else int(self.df["player_id"].iloc[0])
            if resolved_id not in self.df["player_id"].values:
                return pd.DataFrame()
            target = self.df[self.df["player_id"] == resolved_id].iloc[0]

        pool = self.df.copy()

        if exclude_same_team and "team_name" in target:
            pool = pool[pool["team_name"] != target["team_name"]]

        # Filtrar jugadores de la misma posición macro
        target_pos = target["position"]
        pool = pool[pool["position"] == target_pos]

        if len(pool) < 2:
            pool = self.df[self.df["player_id"] != resolved_id].copy()

        features = ["xg_p90", "xa_p90", "interceptions_p90", "tackles_p90", "dribbles_p90", "prog_passes_p90"]
        available_feats = [f for f in features if f in pool.columns]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pool[available_feats].fillna(0))

        # Vector del jugador objetivo
        target_feats_df = target[available_feats].to_frame().T.infer_objects(copy=False).fillna(0)
        target_vector = scaler.transform(target_feats_df)

        k_neighbors = min(k_count + 1, len(pool))
        knn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
        knn.fit(X_scaled)

        distances, indices = knn.kneighbors(target_vector)
        dist_list = distances[0]
        idx_list = indices[0]

        results = []
        for d, idx in zip(dist_list, idx_list):
            cand = pool.iloc[idx]
            if cand["player_id"] == resolved_id:
                continue

            # Convertir distancia euclidiana en porcentaje de similitud (99% = idéntico)
            sim_pct = round(float(np.clip(100.0 - (d * 8.5), 60.0, 99.2)), 1)
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
                "similarity_score": sim_pct,
                "potential_ovr": int(cand.get("potential_ovr", 82)),
                "market_value_m": cand_mv,
                "savings_vs_target_m": saving_m,
                "xg_p90": round(float(cand.get("xg_p90", 0.0)), 2),
                "xa_p90": round(float(cand.get("xa_p90", 0.0)), 2),
                "interceptions_p90": round(float(cand.get("interceptions_p90", 0.0)), 2),
            })

            if len(results) >= k_count:
                break

        return pd.DataFrame(results).sort_values(by="similarity_pct", ascending=False)

    def calculate_moneyball_bargains(self, min_minutes: int = 600, top_n: int = 10) -> pd.DataFrame:
        """
        Calcula el 'Moneyball Index': ratio de rendimiento ofensivo + defensivo por millón de euros.
        Identifica gangas o jugadores infravalorados ideales para clubes con presupuesto ajustado.
        """
        df_eval = self.df[self.df["minutes"] >= min_minutes].copy() if "minutes" in self.df.columns else self.df.copy()
        if df_eval.empty:
            df_eval = self.df.copy()

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

    def generate_pizza_plot(self, player_series: pd.Series):
        """
        Genera un Pizza Plot (gráfico de percentiles circular) estándar de la industria (mplsoccer)
        para el futbolista comparado con su posición en la base de datos de referencia.
        Maneja robustamente porteros y series con valores idénticos o nulos mediante rangos medios (mid-rank).
        """
        try:
            from mplsoccer import PyPizza
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        pos = str(player_series.get("position", "MF")).upper()
        pool = self.df[self.df["position"] == pos]
        if pool.empty or len(pool) < 5:
            pool = self.df

        if "GK" in pos:
            metrics = [
                ("prog_passes_p90", "Juego de Pies"),
                ("interceptions_p90", "Salidas / Centros"),
                ("tackles_p90", "Coberturas"),
                ("minutes", "Minutos"),
                ("market_value_m", "Valor Mercado"),
                ("potential_ovr", "Potencial OVR"),
                ("age", "Experiencia"),
            ]
        else:
            metrics = [
                ("xg_p90", "xG /90"),
                ("xa_p90", "xA /90"),
                ("prog_passes_p90", "Pases Prog"),
                ("dribbles_p90", "Regates /90"),
                ("interceptions_p90", "Intercepciones"),
                ("tackles_p90", "Entradas /90"),
                ("market_value_m", "Valor Mercado"),
                ("potential_ovr", "Potencial OVR")
            ]

        params = []
        values = []
        for col, label in metrics:
            val = float(player_series.get(col, 0.0) or 0.0)
            series = pool[col].dropna() if col in pool.columns else pd.Series([val])
            if not series.empty and len(series) > 0:
                # Fórmula de percentil mid-rank para evitar empates al 100% o al 0%
                strict_less = float((series < val).mean())
                less_equal = float((series <= val).mean())
                pct = int(round(((strict_less + less_equal) / 2.0) * 100))
                pct = int(np.clip(pct, 5, 99))
            else:
                pct = 50
            params.append(label)
            values.append(pct)

        baker = PyPizza(
            params=params,
            background_color="#0b0f19",
            straight_line_color="#334155",
            last_circle_color="#334155",
            other_circle_color="#1e293b",
            inner_circle_size=15
        )

        slice_color = "#38bdf8" if "GK" in pos else "#00ff87"
        fig, ax = baker.make_pizza(
            values,
            figsize=(6.5, 6.5),
            param_location=110,
            kwargs_slices=dict(facecolor=slice_color, edgecolor="#ffffff", linewidth=1.5, zorder=2),
            kwargs_params=dict(color="#ffffff", fontsize=9, va="center"),
            kwargs_values=dict(color="#000000", fontsize=9, zorder=3, bbox=dict(edgecolor="#ffffff", facecolor="#ffffff", boxstyle="round,pad=0.2", lw=1))
        )
        return fig

