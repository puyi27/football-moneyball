from typing import Dict, List, Any, Optional
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
        Garantiza compatibilidad total con la UI (claves 'instruccion_analista' y 'explicacion').
        """
        plays = []

        avg_xg = float(lineup_df.get("xg_p90", pd.Series([0.0])).mean())
        avg_xa = float(lineup_df.get("xa_p90", pd.Series([0.0])).mean())
        avg_int = float(lineup_df.get("interceptions_p90", pd.Series([0.0])).mean())
        avg_tkl = float(lineup_df.get("tackles_p90", pd.Series([0.0])).mean())

        high_line_vuln = float(opponent_profile.get("high_line_vulnerability", 0.5))
        low_block = float(opponent_profile.get("low_block_tendency", 0.5))
        attack_threat = float(opponent_profile.get("attack_threat", 1.0))
        aerial_vuln = float(opponent_profile.get("aerial_weakness", 0.4))
        pressing_int = float(opponent_profile.get("pressing_intensity", 7.0))

        # 1. Transición Rápida y Ataque Vertical al Espacio
        if avg_xg >= 0.20 and high_line_vuln >= 0.60:
            inst = "Atraer en campo propio con primer pase de pivotes y lanzar desmarques de ruptura inmediatos entre central y lateral rival."
            plays.append({
                "categoria": "Transición Ofensiva",
                "jugada": "Transición Vertical y Balones al Espacio a Espalda de Centrales",
                "contexto": (
                    f"El XI cuenta con alta finalización (media {avg_xg:.2f} xG p90) "
                    f"frente a una defensa rival adelantada con vulnerabilidad {high_line_vuln:.2f}."
                ),
                "instruccion_analista": inst,
                "explicacion": inst,
                "impacto_esperado": "Maximización de ocasiones manifiestas (+0.22 xG por tiro en transición).",
            })

        # 2. Sobrecarga Exterior y Pases Atrás (Cutbacks)
        if avg_xa >= 0.15 and low_block >= 0.55:
            inst = "Generar 2vs1 en carril lateral para llegar a línea de fondo y filtrar centro tenso hacia atrás, atacando la zona libre entre mediocentros y defensas rivales."
            plays.append({
                "categoria": "Ataque Posicional",
                "jugada": "Sobrecarga en Bandas y Pase Atrás a Punto de Penalti",
                "contexto": (
                    f"Rival replegado en bloque bajo ({low_block:.2f}) frente a laterales/extremos "
                    f"con capacidad de generación (media {avg_xa:.2f} xA p90)."
                ),
                "instruccion_analista": inst,
                "explicacion": inst,
                "impacto_esperado": "Aumento del volumen de remates dentro del área (+18% probabilidad de gol).",
            })

        # 3. Presión Tras Pérdida y Bloque Medio
        if (avg_int + avg_tkl) >= 1.5 and attack_threat >= 1.15:
            inst = "Realizar acoso intensivo en los primeros 5 segundos tras perder la posesión para cortar la salida limpia y provocar pérdidas en campo contrario."
            plays.append({
                "categoria": "Fase Defensiva",
                "jugada": "Presión Agresiva Tras Pérdida (Gegenpressing en Bloque Medio)",
                "contexto": (
                    f"El rival dispone de gran volumen ofensivo ({attack_threat:.2f}), "
                    f"pero nuestro XI registra un alto promedio de recuperación ({avg_int + avg_tkl:.2f} p90)."
                ),
                "instruccion_analista": inst,
                "explicacion": inst,
                "impacto_esperado": "Neutralización de transiciones rivales y reducción drástica de mu.",
            })

        # 4. Salida de Balón vs Presión Asfixiante (Tercer Hombre)
        if pressing_int >= 7.8:
            inst = "Incrustar mediocentro organizador entre centrales para generar superioridad 3vs2 en primera línea y conectar con el interior libre mediante apoyo en pared."
            plays.append({
                "categoria": "Iniciación de Juego",
                "jugada": "Salida Limpia con Tercer Hombre y Fijación de Primera Línea",
                "contexto": f"Rival con presión muy alta ({pressing_int:.1f}/10); riesgo de pérdida en salida si no se crean líneas de pase diagonales.",
                "instruccion_analista": inst,
                "explicacion": inst,
                "impacto_esperado": "Superación de primera línea de presión rival (+25% progresiones limpias a último tercio).",
            })

        # 5. Balón Parado Ofensivo y Segundas Jugadas (ABP)
        if aerial_vuln >= 0.65:
            inst = "Ejecutar saques de esquina cerrados al primer poste con bloqueo previo al portero y dos jugadores esperando el rechace en la frontal."
            plays.append({
                "categoria": "Balón Parado",
                "jugada": "Córners al Primer Palo y Ocupación del Segundo Rebote",
                "contexto": f"El adversario presenta una debilidad en juego aéreo del {aerial_vuln:.0%}.",
                "instruccion_analista": inst,
                "explicacion": inst,
                "impacto_esperado": "Generación de ocasiones de alto valor xG a balón parado.",
            })

        # 6. Directriz por Defecto si no se activan disparadores extremos
        if not plays:
            inst = "Mantener posesiones largas para gestionar el ritmo y desgastar físicamente al rival."
            plays.append({
                "categoria": "Estrategia Global",
                "jugada": "Circulación de Balón y Posesión Estructurada",
                "contexto": "Equilibrio estadístico general entre las fortalezas de ambos equipos.",
                "instruccion_analista": inst,
                "explicacion": inst,
                "impacto_esperado": "Control territorial y minimización de pérdidas no forzadas.",
            })

        return plays

    def suggest_in_match_adjustments(
        self,
        current_minute: int,
        score_diff: int,
        opponent_profile: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, str]]:
        """
        Prescribe ajustes tácticos en tiempo real desde la banda durante el transcurso del encuentro.
        Ajusta la instrucción según el minuto de partido y la ventaja o desventaja en el marcador.
        """
        adjustments = []
        opp = opponent_profile or {"defense_weakness": 1.0, "attack_threat": 1.0, "pressing_intensity": 7.0}

        if score_diff > 0:
            # Ventaja en el marcador
            if current_minute >= 70:
                inst = "Transición a bloque 5-4-1 compacto. Juntar líneas a 25 metros de portería, forzar al rival a centros frontales y ceder posesión estéril."
                adjustments.append({
                    "minuto_aplicacion": f"{current_minute}'",
                    "estado_partido": f"Defendiendo Ventaja (+{score_diff})",
                    "accion": "Blindaje Defensivo y Gestión de Posesión",
                    "instruccion": inst,
                    "explicacion": inst,
                    "impacto": "Reducción del 40% del xG concedido en el tramo final.",
                })
            else:
                inst = "Mantener ritmo de posesión y pausa con balón. Evitar pérdidas precipitadas en transición ofensiva para frustrar la presión rival."
                adjustments.append({
                    "minuto_aplicacion": f"{current_minute}'",
                    "estado_partido": f"Gestionando Ventaja (+{score_diff})",
                    "accion": "Control Territorial y Pausa",
                    "instruccion": inst,
                    "explicacion": inst,
                    "impacto": "Disminución de transiciones descontroladas.",
                })
        elif score_diff < 0:
            # Desventaja en el marcador
            if current_minute >= 65:
                inst = "Incorporar un segundo punta, adelantar la línea defensiva a campo contrario y ordenar centros inmediatos al área con extremos a pie natural."
                adjustments.append({
                    "minuto_aplicacion": f"{current_minute}'",
                    "estado_partido": f"Buscando Empate ({score_diff})",
                    "accion": "Asedio Ofensivo y Juego Directo",
                    "instruccion": inst,
                    "explicacion": inst,
                    "impacto": "Aumento del 35% en remates generados dentro del área en los últimos 20 minutos.",
                })
            else:
                inst = "Intensificar presión tras pérdida y subir la altura de los laterales para ensanchar el campo y sobrecargar carriles interiores."
                adjustments.append({
                    "minuto_aplicacion": f"{current_minute}'",
                    "estado_partido": f"En Desventaja ({score_diff})",
                    "accion": "Amplitud Total y Sobrecarga Interior",
                    "instruccion": inst,
                    "explicacion": inst,
                    "impacto": "Recuperación de la iniciativa y generación de ocasiones.",
                })
        else:
            # Marcador igualado (empate)
            if current_minute >= 75:
                inst = "Introducir atacantes de refresco por bandas para explotar el cansancio físico de los laterales rivales con desbordes 1vs1."
                adjustments.append({
                    "minuto_aplicacion": f"{current_minute}'",
                    "estado_partido": "Empate en Tramo Decisivo (0)",
                    "accion": "Desequilibrio Exterior con Piernas Frescas",
                    "instruccion": inst,
                    "explicacion": inst,
                    "impacto": "Incremento de probabilidad de gol en los últimos 15 minutos sin desproteger la zaga.",
                })
            else:
                inst = "Mantener balance estructural 4-3-3. Priorizar duelos en la medular y asegurar la vigilancia sobre los contraataques rivales."
                adjustments.append({
                    "minuto_aplicacion": f"{current_minute}'",
                    "estado_partido": "Empate Equilibrado (0)",
                    "accion": "Equilibrio Posicional",
                    "instruccion": inst,
                    "explicacion": inst,
                    "impacto": "Control táctico sin conceder transiciones peligrosas.",
                })

        return adjustments

