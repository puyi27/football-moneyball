from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from src.models_scouting.player_profiler import PlayerProfiler
from src.data.real_dataset_loader import generate_curated_real_dataset


class AcademyScout:
    """
    Ojeador Inteligente de Canteras y Detección de Nuevas Promesas (Wonderkids).
    Permite:
      1. Monitorear la propia cantera del club (La Masia, La Fábrica, Lezama, etc.).
      2. Descubrir 'perlas' en canteras rivales y ligas formadoras (Portugal, Holanda, etc.).
      3. Analizar la curva de progresión: OVR actual vs OVR potencial (Growth Delta).
      4. Recomendar la ruta de desarrollo óptima (Promoción al 1er equipo, Cesión formativa o Fichaje sub-20).
      5. Calcular el impacto financiero: Ahorro en millones de euros y Fair Play Financiero
         al promover un canterano frente a fichar a un jugador consolidado equivalente.
    """

    def __init__(self, players_df: Optional[pd.DataFrame] = None):
        if players_df is not None:
            self.df = players_df.copy()
        else:
            self.df = generate_curated_real_dataset()

        # Asegurar cálculo de OVR actual y asegurar columnas de cantera
        self._ensure_profiling_and_potential()

    def _ensure_profiling_and_potential(self) -> None:
        """Calcula el OVR actual si no está presente y normaliza los potenciales."""
        profiler = PlayerProfiler(self.df)

        ovrs = []
        for _, row in self.df.iterrows():
            if "ovr" in row and not pd.isna(row["ovr"]) and row["ovr"] > 0:
                ovrs.append(int(row["ovr"]))
            else:
                fifa_card = profiler.calculate_fifa_card(row)
                ovrs.append(int(fifa_card.get("OVR", fifa_card.get("ovr", 75))))

        self.df["current_ovr"] = ovrs

        if "potential_ovr" not in self.df.columns:
            self.df["potential_ovr"] = np.nan

        potentials = []
        for _, r in self.df.iterrows():
            pot = r.get("potential_ovr", np.nan)
            if pd.isna(pot) or pot is None or pot <= 0:
                age = float(r.get("age", 25))
                ovr = int(r["current_ovr"])
                if age <= 19:
                    pot = min(94, ovr + 8)
                elif age <= 21:
                    pot = min(92, ovr + 5)
                elif age <= 23:
                    pot = min(90, ovr + 2)
                else:
                    pot = ovr
            potentials.append(int(pot))
        self.df["potential_ovr"] = potentials

        # Columna delta de crecimiento
        self.df["growth_delta"] = self.df["potential_ovr"] - self.df["current_ovr"]
        # Evitar deltas negativos para promesas
        self.df["growth_delta"] = self.df["growth_delta"].apply(lambda d: max(0, int(d)))

        if "is_academy" not in self.df.columns:
            self.df["is_academy"] = False
        else:
            self.df["is_academy"] = self.df["is_academy"].fillna(False).astype(bool)

        if "academy_origin" not in self.df.columns:
            self.df["academy_origin"] = "Fútbol Base Externo"
        else:
            self.df["academy_origin"] = self.df.apply(
                lambda r: str(r["academy_origin"]) if pd.notna(r.get("academy_origin")) and str(r.get("academy_origin")) != "nan" else f"Cantera {r['team_name']}",
                axis=1
            )

    def get_academy_prospects(
        self,
        team_name: Optional[str] = None,
        only_own_academy: bool = False,
        min_potential: int = 80,
        max_age: int = 21,
        position: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filtra y devuelve las promesas de la cantera o de academias rivales.
        """
        filtered = self.df[self.df["potential_ovr"] >= min_potential].copy()
        filtered = filtered[filtered["age"] <= max_age]

        if position:
            filtered = filtered[filtered["position"].str.upper() == position.upper()]

        if team_name:
            if only_own_academy:
                filtered = filtered[filtered["team_name"].str.lower() == team_name.lower()]
            else:
                # Marcar procedencia: propia vs rival
                filtered["is_own_academy"] = filtered["team_name"].str.lower() == team_name.lower()

        # Añadir estimación de ahorro financiero frente a fichaje senior equivalente
        filtered["senior_market_benchmark_m"] = filtered["potential_ovr"].apply(self._estimate_senior_replacement_cost)
        
        # Ahorro neto: Si es propia cantera, el coste de adquisición es 0 M€. Si es rival, es el benchmark - su valor actual
        if team_name and "is_own_academy" in filtered.columns:
            filtered["projected_savings_m"] = np.where(
                filtered["is_own_academy"],
                filtered["senior_market_benchmark_m"],
                np.maximum(0.0, filtered["senior_market_benchmark_m"] - filtered["market_value_m"])
            )
        else:
            filtered["projected_savings_m"] = np.where(
                filtered["is_academy"],
                filtered["senior_market_benchmark_m"],
                np.maximum(0.0, filtered["senior_market_benchmark_m"] - filtered["market_value_m"])
            )

        filtered["projected_savings_m"] = filtered["projected_savings_m"].round(1)

        # Ordenar por potencial descendente y luego delta de crecimiento
        return filtered.sort_values(by=["potential_ovr", "growth_delta"], ascending=[False, False]).reset_index(drop=True)

    def recommend_career_pathway(
        self,
        player_name: str,
        target_team: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genera el informe de ruta de desarrollo (Career Pathway) para una joven promesa.
        """
        matches = self.df[self.df["player_name"].str.lower() == player_name.lower()]
        if matches.empty:
            matches = self.df[self.df["player_name"].str.lower().str.contains(player_name.lower())]
            if matches.empty:
                raise ValueError(f"Jugador '{player_name}' no encontrado en la base de datos de cantera.")

        row = matches.iloc[0]
        age = int(row["age"])
        cur_ovr = int(row["current_ovr"])
        pot_ovr = int(row["potential_ovr"])
        delta = int(row["growth_delta"])
        market_val = float(row.get("market_value_m", 5.0))
        academy = str(row.get("academy_origin", f"Cantera {row.get('team_name', '')}"))
        club = str(row.get("team_name", ""))

        is_own = True
        if target_team and target_team.lower() != club.lower():
            is_own = False

        # 1. Determinar Pathway Estratégico
        if is_own:
            if cur_ovr >= 79 or (cur_ovr >= 75 and age <= 18):
                pathway = "Ascenso Inmediato al Primer Equipo"
                readiness_pct = 90
                action_text = f"Consolidar en la primera plantilla de {club}. Dar minutos de rotación de calidad y titularidades en Copa."
            elif cur_ovr >= 73 and age <= 20:
                pathway = "Cesión Formativa a Primera/Segunda División"
                readiness_pct = 68
                action_text = f"Ceder a un club que garantice 2.000+ minutos en competición oficial (tipo Liga Hypermotion o Eredivisie)."
            else:
                pathway = "Consolidación en Filial B con Dinámica de 1er Equipo"
                readiness_pct = 50
                action_text = "Liderar el filial con entrenamientos semanales bajo supervisión del cuerpo técnico del primer equipo."
        else:
            if pot_ovr >= 87:
                pathway = "Fichaje Estratégico Sub-20 (Oportunidad de Mercado / Poaching)"
                readiness_pct = 82
                action_text = f"Monitorear cláusula de rescisión o fin de contrato en {club}. Perfil generacional con plusvalía garantizada."
            else:
                pathway = "Seguimiento en Lista de Observación (Scouting Activo)"
                readiness_pct = 60
                action_text = f"Enviar ojeador presencial a los próximos 4 partidos de {club} para evaluar toma de decisiones."

        # 2. Análisis de Ahorro e Impacto Fair Play
        senior_cost = self._estimate_senior_replacement_cost(pot_ovr)
        savings = senior_cost if is_own else max(0.0, senior_cost - market_val)

        # 3. Puntos de mejora deportiva
        milestones = self._generate_development_milestones(row)

        return {
            "player_name": row["player_name"],
            "age": age,
            "position": row["position"],
            "club": club,
            "academy_origin": academy,
            "is_own_academy": is_own,
            "current_ovr": cur_ovr,
            "potential_ovr": pot_ovr,
            "growth_delta": delta,
            "market_value_m": market_val,
            "senior_replacement_benchmark_m": round(senior_cost, 1),
            "projected_savings_m": round(savings, 1),
            "recommended_pathway": pathway,
            "readiness_score_pct": readiness_pct,
            "strategic_action": action_text,
            "development_milestones": milestones,
        }

    def compare_academy_vs_market_signing(
        self,
        academy_player_name: str,
        senior_star_name: str
    ) -> Dict[str, Any]:
        """
        Comparador económico y deportivo: Ascender o apostar por la cantera vs Fichar a una estrella consolidada.
        Muestra coste de transferencia, impacto en masa salarial y amortizaciones FFP.
        """
        ac_matches = self.df[self.df["player_name"].str.lower().str.contains(academy_player_name.lower())]
        if ac_matches.empty:
            raise ValueError(f"Canterano '{academy_player_name}' no encontrado.")
        ac_row = ac_matches.iloc[0]

        sr_matches = self.df[self.df["player_name"].str.lower().str.contains(senior_star_name.lower())]
        if sr_matches.empty:
            raise ValueError(f"Jugador senior '{senior_star_name}' no encontrado.")
        sr_row = sr_matches.iloc[0]

        # Costes Cantera
        ac_transfer_cost = 0.0 if ac_row.get("is_academy", True) else float(ac_row.get("market_value_m", 10.0))
        ac_estimated_wage_annual = 1.2  # 1.2M€ sueldo típico canterano primer contrato
        ac_amortization_annual = ac_transfer_cost / 5.0

        # Costes Senior
        sr_transfer_cost = float(sr_row.get("market_value_m", 60.0))
        sr_estimated_wage_annual = max(6.0, round(sr_transfer_cost * 0.12, 1))  # 12% del valor de mercado
        sr_amortization_annual = round(sr_transfer_cost / 4.0, 1)  # Amortización a 4 años

        total_cost_5yr_academy = (ac_transfer_cost + (ac_estimated_wage_annual * 5))
        total_cost_4yr_senior = (sr_transfer_cost + (sr_estimated_wage_annual * 4))

        savings_total = total_cost_4yr_senior - total_cost_5yr_academy

        return {
            "comparison_title": f"Apostar por {ac_row['player_name']} vs Fichar a {sr_row['player_name']}",
            "academy_prospect": {
                "name": ac_row["player_name"],
                "club": ac_row["team_name"],
                "age": int(ac_row["age"]),
                "current_ovr": int(ac_row["current_ovr"]),
                "potential_ovr": int(ac_row["potential_ovr"]),
                "transfer_fee_m": ac_transfer_cost,
                "annual_wage_m": ac_estimated_wage_annual,
                "annual_ffp_impact_m": round(ac_amortization_annual + ac_estimated_wage_annual, 1),
                "total_commitment_m": round(total_cost_5yr_academy, 1),
            },
            "senior_star": {
                "name": sr_row["player_name"],
                "club": sr_row["team_name"],
                "age": int(sr_row["age"]),
                "current_ovr": int(sr_row["current_ovr"]),
                "potential_ovr": int(sr_row["potential_ovr"]),
                "transfer_fee_m": sr_transfer_cost,
                "annual_wage_m": sr_estimated_wage_annual,
                "annual_ffp_impact_m": round(sr_amortization_annual + sr_estimated_wage_annual, 1),
                "total_commitment_m": round(total_cost_4yr_senior, 1),
            },
            "financial_verdict": {
                "net_savings_eur_m": round(savings_total, 1),
                "ffp_relief_annual_m": round((sr_amortization_annual + sr_estimated_wage_annual) - (ac_amortization_annual + ac_estimated_wage_annual), 1),
                "verdict_summary": f"La apuesta por {ac_row['player_name']} libera {round(savings_total, 1)} M€ de presupuesto y salvaguarda el límite salarial de LaLiga."
            }
        }

    @staticmethod
    def _estimate_senior_replacement_cost(potential_ovr: int) -> float:
        """Estima cuánto costaría en el mercado un jugador consolidado con ese OVR."""
        if potential_ovr >= 92:
            return 120.0
        elif potential_ovr >= 89:
            return 85.0
        elif potential_ovr >= 86:
            return 55.0
        elif potential_ovr >= 83:
            return 32.0
        elif potential_ovr >= 80:
            return 18.0
        else:
            return 10.0

    @staticmethod
    def _generate_development_milestones(row: pd.Series) -> List[str]:
        """Genera hitos táctico-físicos personalizados para el plan de trabajo de la promesa."""
        pos = str(row.get("position", "MF")).upper()
        milestones = []
        if "FW" in pos:
            milestones.append("Fase de definición: optimizar ratio disparos entre los 3 palos en transiciones rápidas.")
            milestones.append("Físico: Ganancia de masa muscular en tren inferior para aguantar duelos espaldas a portería.")
        elif "DF" in pos:
            milestones.append("Táctica defensiva: perfilado corporal y anticipación en vigilancias a la espalda de laterales.")
            milestones.append("Salida de balón: batir la primera línea de presión rival con pases tensos interiores.")
        else:
            milestones.append("Visión espacial: mejorar escaneo visual antes de recepción orientada bajo presión alta.")
            milestones.append("Recuperación: intensidad en el 'counter-pressing' en los 5 segundos posteriores a la pérdida.")
        return milestones
