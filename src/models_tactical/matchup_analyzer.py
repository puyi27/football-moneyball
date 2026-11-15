from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class MatchupAnalyzer:
    """
    Optimizador de Alineaciones y Matchups Tácticos frente a rivales específicos.
    Cruza las métricas p90 de la plantilla disponible con las debilidades y fortalezas
    del próximo adversario para seleccionar el XI inicial con mayor probabilidad de éxito.
    """

    DEFAULT_FORMATION = {"GK": 1, "DF": 4, "MF": 3, "FW": 3}

    def __init__(self, league_avg_xg: float = 1.35):
        self.league_avg_xg = league_avg_xg

    def calculate_player_tactical_fit(
        self,
        player_row: pd.Series,
        opponent_profile: Dict[str, float]
    ) -> float:
        """
        Calcula el Matchup Score de un jugador individual contra el perfil del rival.
        Ajusta la ponderación ofensiva vs defensiva según si el rival es vulnerable o dominante.
        """
        opp_def_weakness = opponent_profile.get("defense_weakness", 1.0)
        opp_att_threat = opponent_profile.get("attack_threat", 1.0)
        high_line_vuln = opponent_profile.get("high_line_vulnerability", 0.5)
        low_block = opponent_profile.get("low_block_tendency", 0.5)

        xg = float(player_row.get("xg_p90", 0.0) or 0.0)
        xa = float(player_row.get("xa_p90", 0.0) or 0.0)
        interceptions = float(player_row.get("interceptions_p90", 0.0) or 0.0)
        tackles = float(player_row.get("tackles_p90", 0.0) or 0.0)

        # Contribución ofensiva: Potenciada si el rival tiene debilidad defensiva
        # o si hay espacios (high line con xG de ruptura) o si hay que romper bloques con xA
        off_score = (xg * (1.0 + 0.5 * high_line_vuln) + xa * (1.0 + 0.5 * low_block)) * opp_def_weakness

        # Contribución defensiva: Potenciada si el rival genera alto volumen ofensivo (contención y repliegue)
        def_score = (interceptions * 0.7 + tackles * 0.3) * opp_att_threat

        # Ponderación según la posición del jugador
        pos = str(player_row.get("position", "MF")).upper()
        if "FW" in pos:
            fit_score = (off_score * 0.85) + (def_score * 0.15)
        elif "DF" in pos or "GK" in pos:
            fit_score = (off_score * 0.20) + (def_score * 0.80)
        else:  # Centrocampistas (MF)
            fit_score = (off_score * 0.55) + (def_score * 0.45)

        return float(round(fit_score, 4))

    def recommend_optimal_lineup(
        self,
        available_players_df: pd.DataFrame,
        formation: Optional[Dict[str, int]] = None,
        opponent_profile: Optional[Dict[str, float]] = None,
        require_reliable: bool = True
    ) -> pd.DataFrame:
        """
        Selecciona el mejor XI titular dadas las restricciones de formación y el rival.
        Garantiza una alineación de exactamente 11 futbolistas reales sin duplicidades.
        """
        formation = formation or self.DEFAULT_FORMATION
        opponent_profile = opponent_profile or {"defense_weakness": 1.0, "attack_threat": 1.0}

        df = available_players_df.copy()
        if require_reliable and "is_reliable" in df.columns:
            rel_df = df[df["is_reliable"] == True].copy()
            if not rel_df.empty and len(rel_df) >= sum(formation.values()):
                df = rel_df

        if df.empty:
            raise ValueError("No hay jugadores fiables disponibles para confeccionar la alineación.")

        # Calcular tactical fit score para cada jugador disponible
        df["tactical_fit_score"] = df.apply(
            lambda row: self.calculate_player_tactical_fit(row, opponent_profile), axis=1
        )

        selected_player_ids = set()
        selected_rows = []

        # 1. Asignar por posición preferente sin duplicar futbolistas
        for position, count in formation.items():
            pos_candidates = df[
                (df["position"] == position) & (~df["player_id"].isin(selected_player_ids))
            ].sort_values(by="tactical_fit_score", ascending=False)

            picked = pos_candidates.head(count)
            for _, r in picked.iterrows():
                selected_player_ids.add(r["player_id"])
                selected_rows.append(r)

            shortfall = count - len(picked)
            if shortfall > 0:
                # Buscar alternativas no seleccionadas (evitando GK para posiciones de campo)
                if position != "GK":
                    alt_candidates = df[
                        (df["position"] != "GK") & (~df["player_id"].isin(selected_player_ids))
                    ].sort_values(by="tactical_fit_score", ascending=False)
                else:
                    alt_candidates = df[
                        ~df["player_id"].isin(selected_player_ids)
                    ].sort_values(by="tactical_fit_score", ascending=False)

                alt_picked = alt_candidates.head(shortfall)
                for _, r in alt_picked.iterrows():
                    selected_player_ids.add(r["player_id"])
                    selected_rows.append(r)

        # 2. Garantizar que la alineación total alcance exactamente el número requerido (11)
        total_required = sum(formation.values())
        if len(selected_rows) < total_required and len(df) > len(selected_rows):
            missing_count = total_required - len(selected_rows)
            remaining_pool = df[~df["player_id"].isin(selected_player_ids)].sort_values(
                by="tactical_fit_score", ascending=False
            )
            extra_picked = remaining_pool.head(missing_count)
            for _, r in extra_picked.iterrows():
                selected_player_ids.add(r["player_id"])
                selected_rows.append(r)

        result_df = pd.DataFrame(selected_rows).drop_duplicates(subset=["player_id"])
        return result_df.sort_values(by="tactical_fit_score", ascending=False)

    def analyze_tactical_matchup(
        self,
        lineup_df: pd.DataFrame,
        opponent_profile: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Genera una radiografía del XI frente al rival: ventajas competitivas,
        zonas de riesgo y balance de fuerza.
        """
        avg_off = float(lineup_df.get("xg_p90", pd.Series([0.0])).mean() + lineup_df.get("xa_p90", pd.Series([0.0])).mean())
        avg_def = float(lineup_df.get("interceptions_p90", pd.Series([0.0])).mean())

        opp_def_vuln = opponent_profile.get("defense_weakness", 1.0)
        opp_att = opponent_profile.get("attack_threat", 1.0)

        insights = []
        if avg_off > 0.45 and opp_def_vuln >= 1.1:
            insights.append("Ventaja Ofensiva Clara: Capacidad de generar desequilibrio constante en último tercio.")
        elif avg_off < 0.25 and opp_def_vuln < 0.9:
            insights.append("Riesgo de Atasco: Rival con bloque férreo; se requerirá juego exterior o balón parado.")

        if avg_def >= 1.4 and opp_att >= 1.1:
            insights.append("Bloque Sólido: Nivel de intercepciones óptimo para contener el ataque rival.")
        elif avg_def < 1.0 and opp_att >= 1.2:
            insights.append("Alerta Defensiva: El rival tiene gran potencia ofensiva y el XI concede recuperaciones.")

        return {
            "xi_avg_offensive_threat": round(avg_off, 3),
            "xi_avg_defensive_recovery": round(avg_def, 3),
            "opponent_defense_weakness": round(opp_def_vuln, 3),
            "opponent_attack_threat": round(opp_att, 3),
            "tactical_insights": insights,
        }

    def predict_match_outcome(
        self,
        home_team: str,
        away_team: str,
        local_lineup: Optional[pd.DataFrame] = None,
        rival_lineup: Optional[pd.DataFrame] = None,
        home_advantage: float = 0.20,
        home_elo: Optional[float] = None,
        away_elo: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predice las probabilidades de resultado 1X2 enfrentando las alineaciones de ambos clubes.
        Garantiza que win_home_prob + draw_prob + win_away_prob sumen exactamente 1.0 (100%).
        Utilizado en el laboratorio 'What-If' del cuadro de mando interactivo (app.py).
        """
        if local_lineup is not None and not local_lineup.empty:
            home_off = float(local_lineup.get("xg_p90", pd.Series([0.35])).mean()) * 3.2
            home_def = float(local_lineup.get("interceptions_p90", pd.Series([1.2])).mean())
        else:
            home_off, home_def = 1.35, 1.2

        if rival_lineup is not None and not rival_lineup.empty:
            away_off = float(rival_lineup.get("xg_p90", pd.Series([0.35])).mean()) * 3.0
            away_def = float(rival_lineup.get("interceptions_p90", pd.Series([1.2])).mean())
        else:
            away_off, away_def = 1.15, 1.2

        # Ataque local vs defensa visitante y ataque visitante vs defensa local
        lam = max(0.35, (home_off * (1.35 / max(away_def, 0.4))) + home_advantage)
        mu = max(0.25, (away_off * (1.35 / max(home_def, 0.4))))

        # Ponderación contextual de Elo si está disponible
        h_elo = home_elo if home_elo is not None else kwargs.get("elo_home")
        a_elo = away_elo if away_elo is not None else kwargs.get("elo_away")
        if h_elo is not None and a_elo is not None:
            elo_diff = (float(h_elo) - float(a_elo)) / 400.0
            lam = max(0.30, lam * (1.0 + elo_diff * 0.15))
            mu = max(0.20, mu * (1.0 - elo_diff * 0.15))

        from scipy.stats import poisson
        max_g = 7
        home_p = np.array([poisson.pmf(i, lam) for i in range(max_g)])
        away_p = np.array([poisson.pmf(j, mu) for j in range(max_g)])
        mat = np.outer(home_p, away_p)
        mat = mat / mat.sum()

        win_home = float(np.sum(np.tril(mat, -1)))
        draw = float(np.sum(np.diag(mat)))
        win_away = float(np.sum(np.triu(mat, 1)))

        tot_p = win_home + draw + win_away
        norm_h = win_home / tot_p
        norm_a = win_away / tot_p

        # Normalización estricta: la suma redondeada da exactamente 1.0000
        w_h = round(norm_h, 4)
        w_a = round(norm_a, 4)
        dr = round(1.0 - (w_h + w_a), 4)

        return {
            "win_home_prob": w_h,
            "draw_prob": dr,
            "win_away_prob": w_a,
            "lambda_home": round(lam, 2),
            "lambda_away": round(mu, 2),
            "home_team": home_team,
            "away_team": away_team,
        }

    def simulate_individual_duels(
        self,
        local_lineup: pd.DataFrame,
        rival_lineup: Optional[pd.DataFrame] = None,
        opponent_profile: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Simula los 5 micro-duelos tácticos individuales decisivos entre el XI local y el rival:
          1. Banda Izquierda Ofensiva (Extremo Izquierdo vs Lateral Derecho rival)
          2. Carril Central / Duelo de Gol (Delantero Centro vs Central Marcador rival)
          3. Banda Derecha Ofensiva (Extremo Derecho vs Lateral Izquierdo rival)
          4. La Sala de Máquinas (Mediocentro Organizador vs Pivote Defensivo rival)
          5. Área de Seguridad / Último Bastión (Central Stopper Local vs Delantero Rival)

        Calcula la probabilidad estocástica de éxito en cada duelo, asigna el veredicto
        y genera el análisis cualitativo y su impacto directo en el xG esperado.
        """
        opp_prof = opponent_profile or {
            "defense_weakness": 1.0,
            "attack_threat": 1.0,
            "high_line_vulnerability": 0.5,
            "low_block_tendency": 0.5,
        }

        # Helper para extraer el mejor jugador disponible por posición o métrica
        def _get_player(df: pd.DataFrame, pos_prefix: str, sort_col: str, fallback_idx: int = 0) -> pd.Series:
            if df is None or df.empty:
                return pd.Series({
                    "player_name": f"Jugador Referencia {pos_prefix}",
                    "position": pos_prefix,
                    "xg_p90": 0.25,
                    "xa_p90": 0.20,
                    "interceptions_p90": 1.1,
                    "tackles_p90": 1.0,
                })
            matched = df[df.get("position", pd.Series([""])).astype(str).str.contains(pos_prefix, case=False, na=False)]
            if not matched.empty and sort_col in matched.columns:
                return matched.sort_values(by=sort_col, ascending=False).iloc[0]
            elif not matched.empty:
                return matched.iloc[0]
            elif fallback_idx < len(df):
                return df.iloc[fallback_idx]
            else:
                return df.iloc[0]

        # Jugadores locales clave
        loc_fw = _get_player(local_lineup, "FW", "xg_p90", 0)
        loc_lw = _get_player(local_lineup, "FW", "xa_p90", 1)
        loc_rw = _get_player(local_lineup, "FW", "xg_p90", 2)
        loc_mf = _get_player(local_lineup, "MF", "xa_p90", 3)
        loc_df = _get_player(local_lineup, "DF", "interceptions_p90", 4)

        # Jugadores rivales clave
        if rival_lineup is not None and not rival_lineup.empty:
            riv_cb = _get_player(rival_lineup, "DF", "interceptions_p90", 0)
            riv_rb = _get_player(rival_lineup, "DF", "tackles_p90", 1)
            riv_lb = _get_player(rival_lineup, "DF", "tackles_p90", 2)
            riv_dm = _get_player(rival_lineup, "MF", "interceptions_p90", 3)
            riv_fw = _get_player(rival_lineup, "FW", "xg_p90", 4)
        else:
            # Arquetipos sintéticos basados en el perfil del rival
            def_weak = float(opp_prof.get("defense_weakness", 1.0))
            att_th = float(opp_prof.get("attack_threat", 1.0))
            riv_cb = pd.Series({"player_name": "Central Rival", "position": "DF", "interceptions_p90": max(0.4, 2.0 / def_weak), "tackles_p90": max(0.4, 1.8 / def_weak), "xg_p90": 0.02, "xa_p90": 0.02})
            riv_rb = pd.Series({"player_name": "Lateral Der Rival", "position": "DF", "interceptions_p90": max(0.4, 1.5 / def_weak), "tackles_p90": max(0.4, 1.7 / def_weak), "xg_p90": 0.04, "xa_p90": 0.08})
            riv_lb = pd.Series({"player_name": "Lateral Izq Rival", "position": "DF", "interceptions_p90": max(0.4, 1.5 / def_weak), "tackles_p90": max(0.4, 1.6 / def_weak), "xg_p90": 0.04, "xa_p90": 0.09})
            riv_dm = pd.Series({"player_name": "Pivote Rival", "position": "MF", "interceptions_p90": max(0.5, 1.8 / def_weak), "tackles_p90": max(0.5, 1.9 / def_weak), "xg_p90": 0.08, "xa_p90": 0.15})
            riv_fw = pd.Series({"player_name": "Delantero Rival", "position": "FW", "xg_p90": 0.35 * att_th, "xa_p90": 0.18 * att_th, "interceptions_p90": 0.4, "tackles_p90": 0.3})

        duels = []

        # 1. Duelo Banda Izquierda: Extremo Izquierdo Local vs Lateral Derecho Rival
        lw_att = float(loc_lw.get("xg_p90", 0.0) or 0.0) * 0.4 + float(loc_lw.get("xa_p90", 0.0) or 0.0) * 0.6
        rb_def = float(riv_rb.get("interceptions_p90", 0.0) or 0.0) * 0.5 + float(riv_rb.get("tackles_p90", 0.0) or 0.0) * 0.5
        score_diff_1 = (lw_att * 2.8) - (rb_def * 0.6)
        prob_1 = float(np.clip(1.0 / (1.0 + np.exp(-score_diff_1 * 1.4)), 0.08, 0.92))
        v_1 = "Ventaja Local" if prob_1 >= 0.53 else ("Ventaja Visitante" if prob_1 <= 0.47 else "Duelo Equilibrado")
        duels.append({
            "duel_id": 1,
            "zone": "Flanco Izquierdo Ofensivo",
            "duel_name": "Desborde Exterior vs Repliegue Lateral",
            "local_player": f"{loc_lw.get('player_name', 'Extremo')} ({loc_lw.get('position', 'FW')})",
            "rival_player": f"{riv_rb.get('player_name', 'Lateral')} ({riv_rb.get('position', 'DF')})",
            "local_win_prob": round(prob_1, 3),
            "verdict": v_1,
            "key_insight": (
                f"{loc_lw.get('player_name', 'Atacante')} genera {lw_att:.2f} xG+xA p90; "
                f"{'supera con velocidad la marca exterior rival.' if prob_1 >= 0.5 else 'el lateral rival cierra con solidez la banda.'}"
            )
        })

        # 2. Duelo Carril Central: Delantero Centro Local vs Central Rival
        st_att = float(loc_fw.get("xg_p90", 0.0) or 0.0) * 1.2 + float(loc_fw.get("xa_p90", 0.0) or 0.0) * 0.3
        cb_def = float(riv_cb.get("interceptions_p90", 0.0) or 0.0) * 0.6 + float(riv_cb.get("tackles_p90", 0.0) or 0.0) * 0.4
        score_diff_2 = (st_att * 2.2) - (cb_def * 0.55)
        prob_2 = float(np.clip(1.0 / (1.0 + np.exp(-score_diff_2 * 1.5)), 0.08, 0.92))
        v_2 = "Ventaja Local" if prob_2 >= 0.53 else ("Ventaja Visitante" if prob_2 <= 0.47 else "Duelo Equilibrado")
        duels.append({
            "duel_id": 2,
            "zone": "Carril Central / Último Tercio",
            "duel_name": "Finalización en Área vs Anticipación Central",
            "local_player": f"{loc_fw.get('player_name', 'Delantero')} ({loc_fw.get('position', 'FW')})",
            "rival_player": f"{riv_cb.get('player_name', 'Central')} ({riv_cb.get('position', 'DF')})",
            "local_win_prob": round(prob_2, 3),
            "verdict": v_2,
            "key_insight": (
                f"{loc_fw.get('player_name', 'El 9')} ({st_att:.2f} p90) "
                f"{'tiene ventaja posicional en desmarques al espacio interior.' if prob_2 >= 0.5 else 'enfrenta a un central dominante por alto y en corte bajo.'}"
            )
        })

        # 3. Duelo Banda Derecha: Extremo Derecho Local vs Lateral Izquierdo Rival
        rw_att = float(loc_rw.get("xg_p90", 0.0) or 0.0) * 0.5 + float(loc_rw.get("xa_p90", 0.0) or 0.0) * 0.5
        lb_def = float(riv_lb.get("interceptions_p90", 0.0) or 0.0) * 0.5 + float(riv_lb.get("tackles_p90", 0.0) or 0.0) * 0.5
        score_diff_3 = (rw_att * 2.6) - (lb_def * 0.6)
        prob_3 = float(np.clip(1.0 / (1.0 + np.exp(-score_diff_3 * 1.4)), 0.08, 0.92))
        v_3 = "Ventaja Local" if prob_3 >= 0.53 else ("Ventaja Visitante" if prob_3 <= 0.47 else "Duelo Equilibrado")
        duels.append({
            "duel_id": 3,
            "zone": "Flanco Derecho Ofensivo",
            "duel_name": "Diagonal Interior vs Duelo 1vs1 Lateral",
            "local_player": f"{loc_rw.get('player_name', 'Extremo')} ({loc_rw.get('position', 'FW')})",
            "rival_player": f"{riv_lb.get('player_name', 'Lateral')} ({riv_lb.get('position', 'DF')})",
            "local_win_prob": round(prob_3, 3),
            "verdict": v_3,
            "key_insight": (
                f"Recorte hacia pierna hábil de {loc_rw.get('player_name', 'Extremo')}; "
                f"{'desestabiliza la cobertura exterior del rival.' if prob_3 >= 0.5 else 'el zaguero rival mantiene perfil defensivo cerrado.'}"
            )
        })

        # 4. Duelo Sala de Máquinas: Mediocentro Local vs Pivote / Interior Rival
        mf_play = float(loc_mf.get("xa_p90", 0.0) or 0.0) * 1.2 + float(loc_mf.get("interceptions_p90", 0.0) or 0.0) * 0.6
        dm_play = float(riv_dm.get("interceptions_p90", 0.0) or 0.0) * 0.8 + float(riv_dm.get("tackles_p90", 0.0) or 0.0) * 0.6
        score_diff_4 = (mf_play * 1.8) - (dm_play * 0.9)
        prob_4 = float(np.clip(1.0 / (1.0 + np.exp(-score_diff_4 * 1.3)), 0.08, 0.92))
        v_4 = "Ventaja Local" if prob_4 >= 0.53 else ("Ventaja Visitante" if prob_4 <= 0.47 else "Duelo Equilibrado")
        duels.append({
            "duel_id": 4,
            "zone": "Sala de Máquinas (Medular)",
            "duel_name": "Filtración y Ritmo vs Contención de Bloque",
            "local_player": f"{loc_mf.get('player_name', 'Centrocampista')} ({loc_mf.get('position', 'MF')})",
            "rival_player": f"{riv_dm.get('player_name', 'Pivote')} ({riv_dm.get('position', 'MF')})",
            "local_win_prob": round(prob_4, 3),
            "verdict": v_4,
            "key_insight": (
                f"Batalla en el círculo central: {loc_mf.get('player_name', 'Organizador')} "
                f"{'impone su visión y ritmo de pase.' if prob_4 >= 0.5 else 'sufre ante la presión física rival.'}"
            )
        })

        # 5. Duelo Contención Propia: Central Local vs Delantero Rival
        df_rec = float(loc_df.get("interceptions_p90", 0.0) or 0.0) * 0.6 + float(loc_df.get("tackles_p90", 0.0) or 0.0) * 0.4
        riv_fw_att = float(riv_fw.get("xg_p90", 0.0) or 0.0) * 1.2 + float(riv_fw.get("xa_p90", 0.0) or 0.0) * 0.3
        score_diff_5 = (df_rec * 0.65) - (riv_fw_att * 1.9)
        prob_5 = float(np.clip(1.0 / (1.0 + np.exp(-score_diff_5 * 1.4)), 0.08, 0.92))
        v_5 = "Ventaja Local" if prob_5 >= 0.53 else ("Ventaja Visitante" if prob_5 <= 0.47 else "Duelo Equilibrado")
        duels.append({
            "duel_id": 5,
            "zone": "Área de Contención Local",
            "duel_name": "Blindaje Defensivo vs Remate del Ariete Rival",
            "local_player": f"{loc_df.get('player_name', 'Central')} ({loc_df.get('position', 'DF')})",
            "rival_player": f"{riv_fw.get('player_name', 'Delantero')} ({riv_fw.get('position', 'FW')})",
            "local_win_prob": round(prob_5, 3),
            "verdict": v_5,
            "key_insight": (
                f"{loc_df.get('player_name', 'Central')} ({df_rec:.2f} recuperaciones p90) "
                f"{'neutraliza las recepciones de espaldas del atacante.' if prob_5 >= 0.5 else 'vigilar desmarques al segundo palo del punta rival.'}"
            )
        })

        loc_wins = sum(1 for d in duels if d["verdict"] == "Ventaja Local")
        riv_wins = sum(1 for d in duels if d["verdict"] == "Ventaja Visitante")
        draws = sum(1 for d in duels if d["verdict"] == "Duelo Equilibrado")

        return {
            "total_duels": len(duels),
            "local_duels_won": loc_wins,
            "rival_duels_won": riv_wins,
            "drawn_duels": draws,
            "tactical_balance": (
                "Dominio Local" if loc_wins > riv_wins else
                ("Dominio Visitante" if riv_wins > loc_wins else "Equilibrio Táctico")
            ),
            "projected_xg_influence": round(float((loc_wins - riv_wins) * 0.08), 3),
            "duels": duels,
        }
