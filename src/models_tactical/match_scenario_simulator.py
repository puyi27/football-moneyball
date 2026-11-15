from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from src.models_analytics.poisson_model import DixonColesModel


class MatchScenarioSimulator:
    """
    Simulador de Escenarios en Tiempo Real ('Digital Twin' táctico estilo NFL).
    Simula situaciones de partido minuto a minuto (1-90) comparando:
      1. La táctica y formación elegida por el Míster.
      2. La contra-táctica óptima sugerida por la IA para explotar debilidades del rival.
      3. El comportamiento y fortalezas del equipo rival.
    Entrega el relato de jugadas, línea temporal de xG, mapa de ocasiones por zonas
    y detección de grietas tácticas propias y ajenas.
    """

    def __init__(self, poisson_model: Optional[DixonColesModel] = None):
        self.poisson = poisson_model or DixonColesModel()

    def simulate_situational_match(
        self,
        home_team: str,
        away_team: str,
        coach_formation: str = "4-3-3",
        coach_style: str = "Posesión y Bloque Alto",
        coach_pressing: float = 7.5,
        opponent_profile: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Simula un partido completo minuto a minuto evaluando la interacción
        entre la táctica del míster, la contra-propuesta de la IA y el rival.
        """
        if seed is not None:
            np.random.seed(seed)

        opp = opponent_profile or {
            "defense_weakness": 1.15,
            "attack_threat": 1.10,
            "pressing_intensity": 8.0,
            "high_line_vulnerability": 0.70,
            "style": "Presión Alta y Defensa Adelantada"
        }

        # 1. EVALUAR AJUSTE DE LA TÁCTICA DEL MÍSTER
        # Si el míster juega posesión contra presión asfixiante, riesgo de pérdidas en salida
        mister_off_boost = 1.0
        mister_def_risk = 1.0

        if "Posesión" in coach_style and opp["pressing_intensity"] >= 8.0:
            mister_def_risk += 0.20  # Riesgo de robo rival en tres cuartos
        elif "Contragolpe" in coach_style and opp["high_line_vulnerability"] >= 0.60:
            mister_off_boost += 0.28  # Castigo severo a la espalda rival
        elif "5-" in coach_formation or "3-" in coach_formation:
            mister_def_risk -= 0.15  # Mayor blindaje en transiciones

        # 2. CALCULAR CONTRA-TÁCTICA RECOMENDADA POR LA IA
        ai_recommendation = self._compute_ai_counter_strategy(opp)

        # 3. TASAS DE GOL ESPERADO (LAMBDA Y MU)
        lam_mister = max(0.4, (1.30 * mister_off_boost) * (opp["defense_weakness"]))
        mu_mister = max(0.4, (1.10 * mister_def_risk) * (opp["attack_threat"]))

        lam_ai = max(0.4, (1.30 * ai_recommendation["off_multiplier"]) * (opp["defense_weakness"]))
        mu_ai = max(0.4, (1.10 * ai_recommendation["def_multiplier"]) * (opp["attack_threat"]))

        # 4. SIMULACIÓN MINUTO A MINUTO (90 MINUTOS)
        match_events_mister, xg_timeline_mister, score_mister = self._simulate_90_minutes(
            home_team, away_team, lam_mister, mu_mister, coach_style, opp["style"]
        )
        match_events_ai, xg_timeline_ai, score_ai = self._simulate_90_minutes(
            home_team, away_team, lam_ai, mu_ai, ai_recommendation["style"], opp["style"]
        )

        # 5. DETECCIÓN DE VULNERABILIDADES PROPIAS Y AJENAS
        flaws_detected = self._audit_tactical_vulnerabilities(coach_formation, coach_style, opp)

        return {
            "matchup": f"{home_team} vs {away_team}",
            "coach_tactics": {
                "formation": coach_formation,
                "style": coach_style,
                "pressing": coach_pressing,
                "expected_lambda": round(lam_mister, 2),
                "expected_mu": round(mu_mister, 2),
                "simulated_score": score_mister,
                "events_log": match_events_mister,
                "xg_timeline": xg_timeline_mister,
            },
            "ai_counter_tactics": {
                "recommended_formation": ai_recommendation["formation"],
                "recommended_style": ai_recommendation["style"],
                "tactical_rationale": ai_recommendation["rationale"],
                "expected_lambda": round(lam_ai, 2),
                "expected_mu": round(mu_ai, 2),
                "simulated_score": score_ai,
                "events_log": match_events_ai,
                "xg_timeline": xg_timeline_ai,
            },
            "own_vulnerabilities_alert": flaws_detected,
        }

    def _compute_ai_counter_strategy(self, opp: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula la formación y estilo óptimo para contrarrestar los patrones rivales."""
        if opp.get("high_line_vulnerability", 0.5) >= 0.65:
            return {
                "formation": "5-3-2 Contragolpe",
                "style": "Transición Rápida y Balones al Espacio",
                "rationale": "El rival presiona muy arriba dejando 45 metros a la espalda de sus centrales. Doble punta rápido y bloque medio-bajo para castigar con pases directos.",
                "off_multiplier": 1.35,
                "def_multiplier": 0.80,
            }
        elif opp.get("low_block_tendency", 0.5) >= 0.60:
            return {
                "formation": "4-2-3-1 Sobrecarga",
                "style": "Apertura a Bandas y Pase Atrás (Cutbacks)",
                "rationale": "Rival encerrado en su área. Necesitamos extremos muy abiertos para estirar la zaga y filtrar balones tensos hacia la frontal.",
                "off_multiplier": 1.25,
                "def_multiplier": 0.88,
            }
        else:
            return {
                "formation": "4-3-3 Dinámico",
                "style": "Presión Tras Pérdida y Control en Bloque Medio",
                "rationale": "Rival con fortaleza en el medio campo. Reforzar el triángulo de pivotes para dominar segundas jugadas y evitar contragolpes.",
                "off_multiplier": 1.20,
                "def_multiplier": 0.90,
            }

    def _simulate_90_minutes(
        self,
        home: str,
        away: str,
        lam: float,
        mu: float,
        home_style: str,
        away_style: str
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame, str]:
        """Genera el motor estocástico posesión a posesión de 90 minutos."""
        events = []
        home_goals = 0
        away_goals = 0
        cum_xg_home = 0.0
        cum_xg_away = 0.0

        timeline_data = []

        # Tasa de llegada por minuto (aproximada para 90 min)
        p_home_chance = lam / 90.0 * 2.8
        p_away_chance = mu / 90.0 * 2.8

        zones = ["Banda Izquierda", "Carril Central / Frontal", "Banda Derecha", "Balón Parado / Córner"]

        for minute in range(1, 91):
            # Ocasión de equipo local
            if np.random.rand() < p_home_chance:
                shot_xg = round(float(np.random.beta(2, 6) * 0.75), 2)
                cum_xg_home += shot_xg
                zone = np.random.choice(zones, p=[0.35, 0.40, 0.20, 0.05])

                is_goal = np.random.rand() < shot_xg
                if is_goal:
                    home_goals += 1
                    desc = f"⚽ ¡GOL DE {home}! Remate certero desde {zone} tras jugada generada con {home_style}."
                else:
                    desc = f"Ocasión manifiesta de {home} desde {zone} (xG: {shot_xg:.2f}). Despeja la defensa rival."

                events.append({"minute": minute, "team": home, "zone": zone, "xg": shot_xg, "is_goal": is_goal, "desc": desc})

            # Ocasión de equipo visitante
            if np.random.rand() < p_away_chance:
                shot_xg_away = round(float(np.random.beta(2, 6) * 0.70), 2)
                cum_xg_away += shot_xg_away
                zone_away = np.random.choice(zones, p=[0.25, 0.45, 0.25, 0.05])

                is_goal_away = np.random.rand() < shot_xg_away
                if is_goal_away:
                    away_goals += 1
                    desc_away = f"⚽ ¡GOL DE {away}! Golpe rival por {zone_away} explotando {away_style}."
                else:
                    desc_away = f"Disparo peligroso de {away} en {zone_away} (xG: {shot_xg_away:.2f}). Detiene el guardameta."

                events.append({"minute": minute, "team": away, "zone": zone_away, "xg": shot_xg_away, "is_goal": is_goal_away, "desc": desc_away})

            timeline_data.append({
                "minute": minute,
                "xg_home": round(cum_xg_home, 2),
                "xg_away": round(cum_xg_away, 2),
                "score": f"{home_goals} - {away_goals}"
            })

        score_str = f"{home_goals} - {away_goals}"
        return events, pd.DataFrame(timeline_data), score_str

    def _audit_tactical_vulnerabilities(
        self,
        formation: str,
        style: str,
        opp: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Identifica fallos estructurales en la pizarra propia frente a este rival."""
        flaws = []
        if "4-3-3" in formation and "Posesión" in style and opp.get("pressing_intensity", 7.0) >= 8.0:
            flaws.append({
                "zona": "Salida de Balón Propia",
                "gravedad": "Alta",
                "descripcion": "El rival presiona con 3 efectivos en primer tercio; riesgo crítico de pérdida en iniciación de centrales.",
                "solucion_ia": "Instruir al portero envío largo a la peinada del extremo o incrustar a un mediocentro como tercer central en fase de salida."
            })

        if "Posesión" in style and opp.get("attack_threat", 1.0) >= 1.20:
            flaws.append({
                "zona": "Espalda de Laterales",
                "gravedad": "Media",
                "descripcion": "Laterales proyectados en campo rival conceden autopistas en repliegue ante contragolpes veloces.",
                "solucion_ia": "Pautar repliegue asimétrico: si sube el lateral izquierdo, el derecho se mantiene cerrado junto a centrales."
            })

        if not flaws:
            flaws.append({
                "zona": "Estructura Global",
                "gravedad": "Baja",
                "descripcion": "Estructura posicional equilibrada; vigilar vigilancia en segundas jugadas de córner.",
                "solucion_ia": "Mantener concentración y dos jugadores en rechace perimetral."
            })

        return flaws

    def simulate_match_timeline(
        self,
        home_team: str,
        away_team: str,
        base_elo_home: float = 1600.0,
        base_elo_away: float = 1600.0,
        seed: Optional[int] = None,
        individual_duels: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simulación estocástica minuto a minuto (1-90) para el gemelo digital (Digital Twin).
        Calcula probabilidades de victoria progresivas, métricas xG acumuladas, mapa de ocasiones
        e integra el balance de duelos individuales si están disponibles.
        """
        if seed is not None:
            np.random.seed(seed)

        # Ratio relativo de Elo
        elo_diff = (base_elo_home - base_elo_away) / 400.0
        lam_home = max(0.5, 1.45 + (elo_diff * 0.5))
        lam_away = max(0.4, 1.15 - (elo_diff * 0.4))

        # Modulación contextual si se aportan los resultados de duelos individuales
        if individual_duels and "projected_xg_influence" in individual_duels:
            duel_xg = float(individual_duels["projected_xg_influence"])
            lam_home = max(0.35, lam_home + duel_xg)
            lam_away = max(0.25, lam_away - duel_xg * 0.6)

        events_raw, timeline_df, score_str = self._simulate_90_minutes(
            home_team, away_team, lam_home, lam_away, "Ataque Posicional", "Bloque Medio y Salida"
        )

        h_g, a_g = [int(s.strip()) for s in score_str.split("-")]

        events = []
        cur_home_goals = 0
        cur_away_goals = 0
        base_p_win = 1.0 / (1.0 + 10.0 ** (-elo_diff))

        for ev in events_raw:
            is_goal = ev.get("is_goal", False)
            xg_val = float(ev.get("xg", 0.2))

            if is_goal:
                if ev["team"] == home_team:
                    cur_home_goals += 1
                else:
                    cur_away_goals += 1
                event_type = "Gol ⚽"
            elif xg_val >= 0.38:
                event_type = "Parada Salvadora 🧤"
            else:
                event_type = "Remate Clave 🎯"

            goal_diff = cur_home_goals - cur_away_goals
            min_left = (90 - ev["minute"]) / 90.0
            
            # Dinámica probabilística Dixon-Coles aproximada para tiempo restante
            dyn_prob = np.clip(
                base_p_win + (goal_diff * 0.28) * (1.0 - min_left * 0.45) + (elo_diff * 0.05 * min_left),
                0.02, 0.98
            )

            events.append({
                "minute": int(ev["minute"]),
                "team": ev["team"],
                "event_type": event_type,
                "xg_value": round(xg_val, 2),
                "win_prob_home": f"{round(float(dyn_prob * 100), 1)}%",
                "win_prob_home_float": round(float(dyn_prob), 4),
                "desc": ev.get("desc", ""),
                "zone": ev.get("zone", "")
            })

        total_xg_home = round(float(timeline_df["xg_home"].iloc[-1]) if not timeline_df.empty else 1.2, 2)
        total_xg_away = round(float(timeline_df["xg_away"].iloc[-1]) if not timeline_df.empty else 0.9, 2)

        return {
            "matchup": f"{home_team} vs {away_team}",
            "final_score": (h_g, a_g),
            "score_string": score_str,
            "total_xg_home": total_xg_home,
            "total_xg_away": total_xg_away,
            "events": events,
            "timeline_df": timeline_df
        }

    def simulate_in_game_tactical_adjustment(
        self,
        current_minute: int,
        current_score: Tuple[int, int],
        current_formation: str = "4-3-3",
        current_style: str = "Posesión y Bloque Alto",
        adjusted_formation: str = "5-3-2",
        adjusted_style: str = "Bloque Bajo y Contragolpe Rápido",
        opponent_profile: Optional[Dict[str, Any]] = None,
        base_elo_diff: float = 0.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Simula el impacto de un ajuste táctico en directo ("What-If In-Game") a partir del minuto actual.
        Evalúa si cambiar la formación/estilo protege una ventaja o maximiza la remontada.
        """
        if seed is not None:
            np.random.seed(seed)

        opp = opponent_profile or {
            "defense_weakness": 1.10,
            "attack_threat": 1.10,
            "pressing_intensity": 7.5,
            "high_line_vulnerability": 0.50,
        }

        mins_remaining = max(1, 90 - current_minute)
        time_fraction = mins_remaining / 90.0

        # Baseline: mantener táctica actual
        base_lam = max(0.3, 1.35 * (1.0 + base_elo_diff * 0.15) * opp["defense_weakness"])
        base_mu = max(0.3, 1.15 * (1.0 - base_elo_diff * 0.15) * opp["attack_threat"])

        # Ajustada: nuevo esquema
        adj_off_boost = 1.0
        adj_def_boost = 1.0

        if "5-" in adjusted_formation or "3-" in adjusted_formation:
            adj_def_boost -= 0.22  # Mayor solidez defensiva
            adj_off_boost -= 0.10  # Menor volumen ofensivo
        elif "4-2-4" in adjusted_formation or "3-4-3" in adjusted_formation:
            adj_off_boost += 0.25  # Asedio ofensivo a la desesperada
            adj_def_boost += 0.20  # Mayor exposición a la contra

        if "Contragolpe" in adjusted_style and opp.get("high_line_vulnerability", 0.5) >= 0.6:
            adj_off_boost += 0.18

        adj_lam = max(0.25, base_lam * adj_off_boost)
        adj_mu = max(0.25, base_mu * adj_def_boost)

        # Proyección esperada para los minutos restantes
        base_rem_lam = base_lam * time_fraction
        base_rem_mu = base_mu * time_fraction

        adj_rem_lam = adj_lam * time_fraction
        adj_rem_mu = adj_mu * time_fraction

        # Matriz Dixon-Coles de los minutos restantes
        base_mat = self.poisson.calculate_match_matrix(base_rem_lam, base_rem_mu, rho=-0.05)
        adj_mat = self.poisson.calculate_match_matrix(adj_rem_lam, adj_rem_mu, rho=-0.05)

        h_cur, a_cur = current_score
        lead = h_cur - a_cur

        def _compute_full_probs(mat: np.ndarray, lead_diff: int) -> Dict[str, float]:
            w, d, l = 0.0, 0.0, 0.0
            max_g = mat.shape[0]
            for i in range(max_g):
                for j in range(max_g):
                    p = mat[i, j]
                    final_diff = lead_diff + (i - j)
                    if final_diff > 0:
                        w += p
                    elif final_diff == 0:
                        d += p
                    else:
                        l += p
            tot = max(1e-9, w + d + l)
            return {"win": round(w / tot, 4), "draw": round(d / tot, 4), "loss": round(l / tot, 4)}

        base_probs = _compute_full_probs(base_mat, lead)
        adj_probs = _compute_full_probs(adj_mat, lead)

        delta_win = round(adj_probs["win"] - base_probs["win"], 4)

        if delta_win >= 0.03:
            verdict = f"Ajuste Táctico Recomendado: cambiar a {adjusted_formation} ({adjusted_style}) eleva la probabilidad de ganar en +{round(delta_win * 100, 1)}%."
        elif delta_win <= -0.03:
            verdict = f"Desaconsejado: el cambio a {adjusted_formation} resta estabilidad y reduce la probabilidad de ganar en {round(delta_win * 100, 1)}%."
        else:
            verdict = f"Neutral: impacto estadístico similar. Mantener según el cansancio físico de la plantilla."

        return {
            "current_minute": current_minute,
            "current_score": f"{h_cur} - {a_cur}",
            "minutes_remaining": mins_remaining,
            "baseline_tactics": f"{current_formation} ({current_style})",
            "adjusted_tactics": f"{adjusted_formation} ({adjusted_style})",
            "baseline_win_prob": base_probs["win"],
            "adjusted_win_prob": adj_probs["win"],
            "delta_win_prob": delta_win,
            "baseline_probabilities": base_probs,
            "adjusted_probabilities": adj_probs,
            "tactical_verdict": verdict,
        }
