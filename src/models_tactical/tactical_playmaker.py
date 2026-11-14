from typing import Dict, List, Any
import pandas as pd


class TacticalPlaymaker:
    """
    Motor de Pizarra y Jugadas Tácticas Recomendadas.
    Traduce el perfil estadístico del Once Titular y los déficits del rival en
    planes de partido y directrices tácticas concretas para el cuerpo técnico.
    """

    def suggest_tactical_plays(
        self,
        lineup_df: pd.DataFrame,
        opponent_profile: Dict[str, float]
    ) -> List[Dict[str, str]]:
        """
        Genera directrices tácticas basadas en reglas de inferencia sobre
        métricas agregadas (xG, xA, intercepciones, duelos) y debilidades del rival.
        """
        plays = []

        avg_xg = float(lineup_df.get("xg_p90", pd.Series([0.0])).mean())
        avg_xa = float(lineup_df.get("xa_p90", pd.Series([0.0])).mean())
        avg_int = float(lineup_df.get("interceptions_p90", pd.Series([0.0])).mean())
        avg_tkl = float(lineup_df.get("tackles_p90", pd.Series([0.0])).mean())

        high_line_vuln = opponent_profile.get("high_line_vulnerability", 0.5)
        low_block = opponent_profile.get("low_block_tendency", 0.5)
        attack_threat = opponent_profile.get("attack_threat", 1.0)
        aerial_vuln = opponent_profile.get("aerial_weakness", 0.4)

        # 1. Transición Rápida y Ataque Vertical al Espacio
        if avg_xg >= 0.20 and high_line_vuln >= 0.60:
            plays.append({
                "categoria": "Transición Ofensiva",
                "jugada": "Transición Vertical y Balones al Espacio a Espalda de Centrales",
                "contexto": (
                    f"El XI cuenta con alta finalización (media {avg_xg:.2f} xG p90) "
                    f"frente a una defensa rival adelantada con vulnerabilidad {high_line_vuln:.2f}."
                ),
                "instruccion_analista": (
                    "Atraer en campo propio con primer pase de pivotes y lanzar desmarques "
                    "de ruptura inmediatos entre central y lateral rival."
                ),
                "impacto_esperado": "Maximización de ocasiones manifiestas (+0.22 xG por tiro en transición).",
            })

        # 2. Sobrecarga Exterior y Pases Atrás (Cutbacks)
        if avg_xa >= 0.15 and low_block >= 0.55:
            plays.append({
                "categoria": "Ataque Posicional",
                "jugada": "Sobrecarga en Bandas y Pase Atrás a Punto de Penalti",
                "contexto": (
                    f"Rival replegado en bloque bajo ({low_block:.2f}) frente a laterales/extremos "
                    f"con capacidad de generación (media {avg_xa:.2f} xA p90)."
                ),
                "instruccion_analista": (
                    "Generar 2vs1 en carril lateral para llegar a línea de fondo y filtrar centro tenso hacia atrás, "
                    "atacando la zona libre entre mediocentros y defensas rivales."
                ),
                "impacto_esperado": "Aumento del volumen de remates dentro del área (+18% probabilidad de gol).",
            })

        # 3. Presión Tras Pérdida y Bloque Medio
        if (avg_int + avg_tkl) >= 1.5 and attack_threat >= 1.15:
            plays.append({
                "categoria": "Fase Defensiva",
                "jugada": "Presión Agresiva Tras Pérdida (Gegenpressing en Bloque Medio)",
                "contexto": (
                    f"El rival dispone de gran volumen ofensivo ({attack_threat:.2f}), "
                    f"pero nuestro XI registra un alto promedio de recuperación ({avg_int + avg_tkl:.2f} p90)."
                ),
                "instruccion_analista": (
                    "Realizar acoso intensivo en los primeros 5 segundos tras perder la posesión "
                    "para cortar la salida limpia y provocar pérdidas en campo contrario."
                ),
                "impacto_esperado": "Neutralización de transiciones rivales y reducción drástica de mu.",
            })

        # 4. Balón Parado Ofensivo y Segundas Jugadas
        if aerial_vuln >= 0.65:
            plays.append({
                "categoria": "Balón Parado",
                "jugada": "Córners al Primer Palo y Ocupación del Segundo Rebote",
                "contexto": f"El adversario presenta una debilidad en juego aéreo del {aerial_vuln:.0%}.",
                "instruccion_analista": (
                    "Ejecutar saques de esquina cerrados al primer poste con bloqueo previo "
                    "al portero y dos jugadores esperando el rechace en la frontal."
                ),
                "impacto_esperado": "Generación de ocasiones de alto valor xG a balón parado.",
            })

        # 5. Directriz por Defecto si no se activan disparadores extremos
        if not plays:
            plays.append({
                "categoria": "Estrategia Global",
                "jugada": "Circulación de Balón y Posesión Estructurada",
                "contexto": "Equilibrio estadístico general entre las fortalezas de ambos equipos.",
                "instruccion_analista": "Mantener posesiones largas para gestionar el ritmo y desgastar físicamente al rival.",
                "impacto_esperado": "Control territorial y minimización de pérdidas no forzadas.",
            })

        return plays
