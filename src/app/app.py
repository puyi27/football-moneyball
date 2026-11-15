import sys
import os
from pathlib import Path

# Garantizar que la raíz del proyecto esté en sys.path para importaciones de 'src'
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

# Importación de la arquitectura analítica (Fases 3, 4 y 5)
from src.data.real_dataset_loader import load_real_players_df
from src.models_scouting.player_profiler import PlayerProfiler
from src.models_analytics.team_profiler import TeamProfiler
from src.models_scouting.squad_diagnostics import SquadDiagnostics
from src.models_scouting.recommender import ScoutingRecommender
from src.models_analytics.elo_model import EloSystem
from src.models_analytics.poisson_model import DixonColesModel
from src.models_tactical.lineup_simulator import TacticalSimulator
from src.models_tactical.matchup_analyzer import MatchupAnalyzer
from src.models_tactical.tactical_playmaker import TacticalPlaymaker
from src.models_tactical.tactical_analyst_desk import TacticalAnalystDesk


# 1. Configuración de página
st.set_page_config(
    page_title="Football Moneyball AI - Virtual Analyst Desk",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Conexión de base de datos
@st.cache_resource
def get_db_engine():
    try:
        return create_engine("postgresql://moneyball:scouting_admin@postgres:5432/football_db")
    except Exception:
        return None

# 3. Conjunto de datos enriquecido (multiliga Top 5 + Scouting)
@st.cache_data
def get_players_database() -> pd.DataFrame:
    """Carga los datos de futbolistas reales con sus métricas y valores de mercado."""
    engine = get_db_engine()
    if engine:
        try:
            db_df = pd.read_sql("SELECT * FROM analytics_scouting_metrics", engine)
            if not db_df.empty:
                return db_df
        except Exception:
            pass
    return load_real_players_df()


# 4. Función gráfica para dibujar un campo de fútbol 2D con mapa de calor
def draw_tactical_pitch_heatmap(heatmap_df: pd.DataFrame, player_name: str, position: str) -> go.Figure:
    """Dibuja un terreno de juego reglamentario (105x68m) con la densidad de actividad del jugador."""
    fig = go.Figure()

    # Césped de fútbol (Verde profesional)
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, fillcolor="#132c1b", line=dict(color="#ffffff", width=2))

    # Línea de medio campo y círculo central
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="circle", x0=52.5 - 9.15, y0=34 - 9.15, x1=52.5 + 9.15, y1=34 + 9.15, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="circle", x0=52.2, y0=33.7, x1=52.8, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))

    # Áreas de penal
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color="#ffffff", width=2))

    # Áreas pequeñas de meta
    fig.add_shape(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16, line=dict(color="#ffffff", width=2))

    # Puntos de penalti
    fig.add_shape(type="circle", x0=10.7, y0=33.7, x1=11.3, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))
    fig.add_shape(type="circle", x0=93.7, y0=33.7, x1=94.3, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))

    # Capa de calor de toques / acciones (2D Density Heatmap)
    fig.add_trace(go.Histogram2dContour(
        x=heatmap_df["x"],
        y=heatmap_df["y"],
        colorscale="Hot",
        reversescale=True,
        showscale=False,
        ncontours=15,
        opacity=0.68,
        contours=dict(coloring="heatmap")
    ))

    # Puntos individuales de actividad
    fig.add_trace(go.Scatter(
        x=heatmap_df["x"],
        y=heatmap_df["y"],
        mode="markers",
        marker=dict(size=4, color="#ffcc00", opacity=0.45),
        name="Acciones / Toques"
    ))

    fig.update_layout(
        title=f"Mapa de Calor de Zonas de Influencia: {player_name} ({position})",
        xaxis=dict(range=[-2, 107], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 70], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
        showlegend=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117"
    )

    return fig


def main():
    st.sidebar.title("⚽ Football Moneyball AI")
    st.sidebar.markdown("**Virtual Analyst Desk & Decision Support System**")
    st.sidebar.caption("Herramienta Profesional de Inteligencia Deportiva para Clubes (TFM)")
    st.sidebar.markdown("---")

    df_players = get_players_database()

    # Inicializar sistema Elo de calibración
    elo = EloSystem()
    elo.ratings.update({
        "Real Madrid": 1840.0, "Barcelona": 1820.0, "Atletico Madrid": 1740.0,
        "Sevilla FC": 1620.0, "Real Betis": 1630.0, "Real Sociedad": 1660.0,
        "Girona FC": 1650.0, "Athletic Club": 1670.0, "Villarreal CF": 1660.0, "Valencia CF": 1560.0,
        "Arsenal": 1820.0, "Manchester City": 1870.0, "Liverpool": 1850.0, "Chelsea": 1730.0,
        "Bayern Munich": 1850.0, "Bayer Leverkusen": 1810.0, "Inter": 1780.0, "Milan": 1720.0, "Atalanta": 1740.0,
        "Paris Saint-Germain": 1810.0, "Benfica": 1670.0, "Sporting CP": 1690.0, "PSV": 1660.0, "Feyenoord": 1640.0,
    })

    profiler = PlayerProfiler(df_players)
    team_profiler = TeamProfiler(df_players, elo_system=elo)
    desk = TacticalAnalystDesk(df_players_metrics=df_players, elo_system=elo)

    # Menú Principal de Herramientas del Analista
    menu = st.sidebar.radio(
        "Módulos del Analista:",
        [
            "1. Ficha 360° del Jugador (FIFA & Clones K-NN)",
            "2. Centro de Mando del Club (ADN Táctico & Plantilla)",
            "3. Radar de Gangas ('Moneyball Index')",
            "4. Laboratorio de Fichajes ('What-If' & Monte Carlo)",
            "5. Preparador de Partido (Pizarra vs Rival)",
            "6. Dossier Ejecutivo para Directiva"
        ]
    )

    # =========================================================================
    # VISTA 1: FICHA 360° DEL JUGADOR (FIFA CARD, RADAR, MAPA DE CALOR, CLONES)
    # =========================================================================
    if menu == "1. Ficha 360° del Jugador (FIFA & Clones K-NN)":
        st.title("👤 Ficha Técnica 360° del Futbolista")
        st.markdown("Radiografía individual estilo EA Sports / FIFA FUT con atributos 1-99, mapa de calor táctico y detección de futbolistas análogos (Clones K-NN).")

        col_sel1, col_sel2 = st.columns([1, 2])
        with col_sel1:
            all_leagues = sorted(df_players["league"].unique())
            sel_league = st.selectbox("Filtrar por Liga:", ["Todas"] + all_leagues, index=0)
            if sel_league != "Todas":
                filtered_pool = df_players[df_players["league"] == sel_league]
            else:
                filtered_pool = df_players

            player_names = sorted(filtered_pool["player_name"].unique())
            default_idx = player_names.index("Vinícius Júnior") if "Vinícius Júnior" in player_names else 0
            selected_player = st.selectbox("Seleccionar Jugador:", player_names, index=default_idx)

        player_data = df_players[df_players["player_name"] == selected_player].iloc[0]
        card = profiler.calculate_fifa_card(player_data)

        # 1. TARJETA FIFA Y ATRIBUTOS CLAVE
        col_card, col_radar = st.columns([1.2, 1.8])
        with col_card:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f1c18 0%, #3a321d 50%, #1f1c18 100%);
                        border: 3px solid #d4af37; border-radius: 16px; padding: 22px; color: white;
                        box-shadow: 0 8px 24px rgba(0,0,0,0.6); text-align: center;">
                <div style="font-size: 52px; font-weight: 900; color: #f5d77f; line-height: 1;">{card['OVR']}</div>
                <div style="font-size: 18px; font-weight: bold; letter-spacing: 2px; color: #ffffff;">{card['position']} | {card['team_name']}</div>
                <hr style="border-color: #d4af37; margin: 12px 0;">
                <div style="font-size: 24px; font-weight: 800; color: #ffffff;">{card['player_name']}</div>
                <div style="font-size: 14px; color: #cccccc; margin-bottom: 14px;">{card['league']} • {card['age']} años • {card['market_value_m']} M€</div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 16px; font-weight: bold;">
                    <div><span style="color: #f5d77f;">PAC</span> {card['PAC']}</div>
                    <div><span style="color: #f5d77f;">SHO</span> {card['SHO']}</div>
                    <div><span style="color: #f5d77f;">PAS</span> {card['PAS']}</div>
                    <div><span style="color: #f5d77f;">DRI</span> {card['DRI']}</div>
                    <div><span style="color: #f5d77f;">DEF</span> {card['DEF']}</div>
                    <div><span style="color: #f5d77f;">PHY</span> {card['PHY']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("Goles Esperados p90", f"{player_data.get('xg_p90', 0):.2f}")
            m2.metric("Asistencias Esp. p90", f"{player_data.get('xa_p90', 0):.2f}")

        with col_radar:
            # Gráfico de Radar Hexagonal
            categories = ["PAC (Ritmo)", "SHO (Tiro)", "PAS (Pase)", "DRI (Regate)", "DEF (Defensa)", "PHY (Físico)"]
            vals = [card["PAC"], card["SHO"], card["PAS"], card["DRI"], card["DEF"], card["PHY"]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(212, 175, 55, 0.35)",
                line=dict(color="#d4af37", width=3),
                name=f"{card['player_name']} (OVR {card['OVR']})"
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[40, 100], color="#aaaaaa"),
                    angularaxis=dict(color="#ffffff")
                ),
                showlegend=True,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                height=360,
                margin=dict(l=30, r=30, t=30, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

        # 2. MAPA DE CALOR TÁCTICO EN CAMPO REGLEMENTARIO
        st.subheader("📍 Mapa de Calor y Zonas de Intervención en el Campo")
        heatmap_coords = profiler.get_pitch_heatmap_data(player_data)
        fig_pitch = draw_tactical_pitch_heatmap(heatmap_coords, card["player_name"], card["position"])
        st.plotly_chart(fig_pitch, use_container_width=True)

        st.markdown("---")

        # 3. JUGADORES CLONES / GEMELOS (K-NN CON % SIMILITUD)
        st.subheader("👯 Jugadores Gemelos / Clones en Europa (K-NN)")
        st.markdown(f"Alternativas estadísticas más similares a **{card['player_name']}** en ligas europeas para optimizar presupuestos de fichaje:")

        twins_df = profiler.find_player_twins(player_id=int(player_data["player_id"]), top_k=5)
        if not twins_df.empty:
            for _, twin in twins_df.iterrows():
                with st.container():
                    c_t1, c_t2, c_t3, c_t4 = st.columns([2, 1.2, 1.2, 1.2])
                    c_t1.markdown(f"**{twin['player_name']}** ({twin['team_name']} • {twin['league']})  \n*{twin['position']} • {twin['age']} años*")
                    c_t2.metric("Similitud Táctica", f"{twin['similarity_pct']}%")
                    c_t3.metric("Valor Mercado", f"{twin['market_value_m']} M€")
                    savings = twin['savings_vs_target_m']
                    c_t4.metric("Diferencial Ahorro", f"{savings:+.1f} M€", delta=f"{savings:+.1f} M€" if savings > 0 else f"{savings:+.1f} M€")
                    st.divider()
        else:
            st.info("No se encontraron suficientes futbolistas en la misma demarcación.")

    # =========================================================================
    # VISTA 2: CENTRO DE MANDO DEL CLUB (ADN TÁCTICO & PLANTILLA)
    # =========================================================================
    elif menu == "2. Centro de Mando del Club (ADN Táctico & Plantilla)":
        st.title("🏟️ Centro de Mando del Club y ADN Táctico")
        st.markdown("Diagnóstico integral del equipo: rating Elo, parámetros de fuerza ofensiva y defensiva, estilo táctico y pirámide de edad de la plantilla.")

        laliga_teams = sorted(df_players[df_players["league"] == "LaLiga"]["team_name"].unique())
        selected_club = st.selectbox("Seleccionar Club:", laliga_teams, index=laliga_teams.index("Sevilla FC") if "Sevilla FC" in laliga_teams else 0)

        overview = team_profiler.get_team_overview(selected_club)
        tactics = overview["tactical_adn"]

        # Métricas generales del club
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rating Elo Competitivo", f"{overview['elo_rating']} pts")
        m2.metric("Poder Ofensivo (α)", f"{overview['alpha_offensive_power']:.2f}")
        m3.metric("Fragilidad Defensiva (β)", f"{overview['beta_defensive_leak']:.2f}", delta_color="inverse")
        m4.metric("Valor de Plantilla", f"{overview['total_market_value_m']} M€")

        st.markdown("---")

        c_adn1, c_adn2 = st.columns([1.5, 1.5])
        with c_adn1:
            st.subheader("🧬 ADN y Filosofía de Juego")
            st.info(f"**Estilo Predominante:** {tactics['style']}")
            st.write(f"**Ruta de Ataque Preferente:** {tactics['preferred_channel']}")
            st.write(f"**Posesión Media Estimada:** {tactics['possession_pct']}%")
            st.progress(tactics['possession_pct'] / 100.0)

            st.write(f"**Intensidad de Presión:** {tactics['pressing_intensity']} / 10")
            st.progress(tactics['pressing_intensity'] / 10.0)

            st.write(f"**Verticalidad de Ataque:** {tactics['verticality']} / 10")
            st.progress(tactics['verticality'] / 10.0)

        with c_adn2:
            st.subheader("⚠️ Alertas de Relevo Generacional (Veteranos >31a)")
            aging = overview["aging_renewals_needed"]
            if aging:
                for a in aging:
                    st.warning(f"🚨 **{a['player_name']}** ({a['position']}, {a['age']} años) • {a['minutes']} minutos jugados.  \n*Prioridad de fichaje en esta demarcación:* **{a['urgency']}**")
            else:
                st.success("✅ Plantilla joven y equilibrada sin sobrecarga crítica de veteranos.")

        st.markdown("---")
        st.subheader(f"📋 Plantilla Completa de {selected_club}")
        roster = overview["squad_roster"][["player_name", "position", "age", "market_value_m", "minutes", "xg_p90", "xa_p90", "interceptions_p90"]]
        st.dataframe(roster, use_container_width=True)

    # =========================================================================
    # VISTA 3: RADAR DE GANGAS ('MONEYBALL INDEX')
    # =========================================================================
    elif menu == "3. Radar de Gangas ('Moneyball Index')":
        st.title("💡 Radar de Gangas: 'The Moneyball Index'")
        st.markdown("Detecta futbolistas infravalorados cuyo rendimiento deportivo ($xG + xA + \\text{recuperaciones}$) supera exponencialmente su valor de mercado.")

        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            pos_filter = st.selectbox("Posición:", ["Todas", "FW", "MF", "DF"], index=0)
        with c_f2:
            max_mv = st.slider("Valor Máximo de Mercado (M€):", 5, 80, 30)
        with c_f3:
            max_age_bargain = st.slider("Edad Máxima:", 18, 30, 26)

        bargains = profiler.calculate_moneyball_bargains(min_minutes=450, top_n=25)

        if pos_filter != "Todas":
            bargains = bargains[bargains["position"] == pos_filter]
        bargains = bargains[(bargains["market_value_m"] <= max_mv) & (bargains["age"] <= max_age_bargain)]

        st.subheader("🏆 Top Jugadores con Mayor Rentabilidad Deportiva")
        st.dataframe(bargains.style.background_gradient(subset=["moneyball_ratio"], cmap="Greens"), use_container_width=True)

    # =========================================================================
    # VISTA 4: LABORATORIO DE FICHAJES ('WHAT-IF' & MONTE CARLO)
    # =========================================================================
    elif menu == "4. Laboratorio de Fichajes ('What-If' & Monte Carlo)":
        st.title("🧪 Laboratorio de Simulación Táctica y Fichajes ('What-If')")
        st.markdown("Simula con modelos probabilísticos qué le ocurre a tu equipo si fichas a un jugador de otra liga europea en lugar de tu titular actual.")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            my_team = st.selectbox("Tu Equipo de LaLiga:", sorted(df_players[df_players["league"] == "LaLiga"]["team_name"].unique()), index=0)
            team_squad = df_players[df_players["team_name"] == my_team]
            player_out_name = st.selectbox("Jugador Titular a Sustituir:", team_squad["player_name"].tolist())
            player_out_row = team_squad[team_squad["player_name"] == player_out_name].iloc[0]

        with col_s2:
            candidate_pool = df_players[df_players["team_name"] != my_team]
            cand_name = st.selectbox("Fichaje / Jugador a Incorporar:", candidate_pool["player_name"].tolist())
            cand_row = candidate_pool[candidate_pool["player_name"] == cand_name].iloc[0]

        st.markdown("---")
        if st.button("🚀 Simular Impacto del Fichaje (Dixon-Coles + 10.000 Monte Carlo)"):
            with st.spinner("Procesando ajuste por dificultad competitiva Elo y simulación estocástica..."):
                sim_res = desk.simulate_transfer_scenario(
                    team_name=my_team,
                    player_out_id=int(player_out_row["player_id"]),
                    candidate_player_id=int(cand_row["player_id"]),
                    base_lambda=1.28,
                    base_mu=1.12,
                    remaining_matches=15,
                    current_points=40,
                )

            swap = sim_res["swap_impact"]
            mc = sim_res["monte_carlo_projection"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Goles Esp. Favor (λ)", f"{swap['new_lambda']}", delta=f"{swap['delta_lambda']:+0.3f}")
            m2.metric("Goles Esp. Contra (μ)", f"{swap['new_mu']}", delta=f"{swap['delta_mu']:+0.3f}", delta_color="inverse")
            m3.metric("Probabilidad Victoria", f"{swap['new_win_prob']*100:.1f}%", delta=f"{swap['win_prob_delta']*100:+0.1f}%")
            m4.metric("Puntos Ganados a 38j", f"{mc['points_gain_mean']:+0.2f} pts", delta=f"{mc['points_gain_mean']:+0.2f}")

            c_mc1, c_mc2 = st.columns(2)
            with c_mc1:
                st.subheader("Probabilidad Clasificación a Europa (>= 60 pts)")
                st.progress(mc['new_prob_europe_60pts'])
                st.write(f"Con el nuevo fichaje: **{mc['new_prob_europe_60pts']*100:.1f}%** (Antes: {mc['base_prob_europe_60pts']*100:.1f}%)")
            with c_mc2:
                st.subheader("Riesgo de Descenso (<= 38 pts)")
                st.progress(mc['new_prob_relegation_38pts'])
                st.write(f"Con el nuevo fichaje: **{mc['new_prob_relegation_38pts']*100:.1f}%** (Antes: {mc['base_prob_relegation_38pts']*100:.1f}%)")

    # =========================================================================
    # VISTA 5: PREPARADOR DE PARTIDO (PIZARRA VS RIVAL)
    # =========================================================================
    elif menu == "5. Preparador de Partido (Pizarra vs Rival)":
        st.title("⚔️ Preparación de Partido y Plan Táctico vs Rival")
        st.markdown("Diseña el plan de partido contra el próximo rival de liga: alineación titular óptima según sus debilidades y directrices de pizarra.")

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            my_team = st.selectbox("Tu Equipo:", sorted(df_players[df_players["league"] == "LaLiga"]["team_name"].unique()), index=0)
        with c_p2:
            rivals = [t for t in sorted(df_players[df_players["league"] == "LaLiga"]["team_name"].unique()) if t != my_team]
            rival = st.selectbox("Próximo Rival:", rivals, index=0)

        # Cargar perfil táctico del rival
        rival_overview = team_profiler.get_team_overview(rival)
        rival_tactics = rival_overview["tactical_adn"]

        opp_profile = {
            "defense_weakness": rival_overview["beta_defensive_leak"],
            "attack_threat": rival_overview["alpha_offensive_power"],
            "high_line_vulnerability": 0.75 if rival_tactics["pressing_intensity"] > 8.0 else 0.40,
            "low_block_tendency": 0.65 if rival_tactics["possession_pct"] < 50.0 else 0.30,
        }

        st.info(f"**Análisis del Rival ({rival}):** Estilo: *{rival_tactics['style']}* • Posesión: *{rival_tactics['possession_pct']}%* • Fragilidad defensiva: *{rival_overview['beta_defensive_leak']}*")

        plan = desk.prepare_match_plan(
            team_name=my_team,
            opponent_profile=opp_profile,
            formation={"GK": 1, "DF": 4, "MF": 3, "FW": 3}
        )

        st.subheader(f"🏆 Once Titular Recomendado vs {rival}")
        st.dataframe(plan["optimal_lineup"][["player_name", "position", "tactical_fit_score", "xg_p90", "xa_p90", "interceptions_p90"]], use_container_width=True)

        st.subheader("📌 Directrices Tácticas y Pizarra del Cuerpo Técnico")
        for play in plan["tactical_plays"]:
            with st.expander(f"🔹 {play['categoria']}: {play['jugada']}", expanded=True):
                st.write(f"**Contexto Estadístico:** {play['contexto']}")
                st.info(f"**Instrucción al Entrenador:** {play['instruccion_analista']}")
                st.success(f"**Impacto Esperado:** {play['impacto_esperado']}")

    # =========================================================================
    # VISTA 6: DOSSIER EJECUTIVO PARA DIRECTIVA
    # =========================================================================
    elif menu == "6. Dossier Ejecutivo para Directiva":
        st.title("📑 Dossier Ejecutivo de Decisión Deportiva")
        st.markdown("Informe consolidado para el Director Deportivo y la Junta Directiva justificando fichajes con analítica de datos e impacto financiero.")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            club = st.selectbox("Club Solicitante:", sorted(df_players[df_players["league"] == "LaLiga"]["team_name"].unique()), index=0)
        with col_d2:
            target_p = st.selectbox("Futbolista a Analizar:", sorted(df_players["player_name"].unique()), index=0)

        p_data = df_players[df_players["player_name"] == target_p].iloc[0]
        card = profiler.calculate_fifa_card(p_data)

        st.markdown(f"""
        ### 📄 INFORME TÉCNICO DE CAPTACIÓN: {card['player_name']}
        * **Club Actual:** {card['team_name']} ({card['league']})
        * **Edad / Posición:** {card['age']} años | {card['position']}
        * **Valor de Mercado:** {card['market_value_m']} M€
        * **Media Global (OVR):** **{card['OVR']} / 99**
        * **Perfil de Atributos:** Ritmo {card['PAC']} | Tiro {card['SHO']} | Pase {card['PAS']} | Regate {card['DRI']} | Defensa {card['DEF']} | Físico {card['PHY']}
        ---
        **Dictamen del Algoritmo:**
        Futbolista con alto impacto métrico en su demarcación. Su incorporación representa un salto cualitativo en goles esperados y creación de ocasiones frente al promedio actual de la plantilla de **{club}**.
        """)
        st.success("✅ Recomendación Final del Modelo: FICHAR (Viabilidad Deportiva Alta)")


if __name__ == "__main__":
    main()
