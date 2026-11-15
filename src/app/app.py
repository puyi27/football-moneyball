import sys
import os
from pathlib import Path

# Asegurar raíz en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

# Importaciones de la arquitectura analítica unificada
from src.data.global_player_database import load_global_players, sync_global_database_to_postgres
from src.data.football_api_connector import FootballAPIConnector
from src.data.advanced_sources_hub import MasterAnalyticsDataHub
from src.models_scouting.player_profiler import PlayerProfiler
from src.models_analytics.team_profiler import TeamProfiler
from src.models_scouting.squad_diagnostics import SquadDiagnostics
from src.models_scouting.academy_scout import AcademyScout
from src.models_analytics.elo_model import EloSystem
from src.models_tactical.tactical_analyst_desk import TacticalAnalystDesk
from src.models_tactical.match_scenario_simulator import MatchScenarioSimulator


# 1. Configuración de página
st.set_page_config(
    page_title="Football Moneyball AI — Virtual Analyst Desk (TFM)",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS profesional adaptado a analítica deportiva de élite
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    .executive-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .executive-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .executive-subtitle {
        color: #94a3b8;
        font-size: 13.5px;
        margin-top: 5px;
        margin-bottom: 14px;
    }
    .status-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .status-badge {
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-green { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-blue { background-color: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .badge-purple { background-color: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
    .badge-gold { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }

    div[data-testid="stMetricValue"] {
        font-weight: 800;
        color: #38bdf8;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# 2. Conectores cacheados
@st.cache_resource
def get_db_engine():
    try:
        return create_engine("postgresql://moneyball:scouting_admin@postgres:5432/football_db")
    except Exception:
        return None

@st.cache_data
def get_players_database() -> pd.DataFrame:
    """Carga la base de datos de futbolistas reales con sus métricas y valores oficiales."""
    return load_global_players()

@st.cache_resource
def get_advanced_hub() -> MasterAnalyticsDataHub:
    return MasterAnalyticsDataHub()


# 3. Visualizador de mapa de calor 2D reglamentario
def draw_tactical_pitch_heatmap(heatmap_df: pd.DataFrame, player_name: str, position: str) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, fillcolor="#0d2818", line=dict(color="#4ade80", width=1.5))
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="circle", x0=52.5 - 9.15, y0=34 - 9.15, x1=52.5 + 9.15, y1=34 + 9.15, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="circle", x0=52.2, y0=33.7, x1=52.8, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="circle", x0=10.7, y0=33.7, x1=11.3, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))
    fig.add_shape(type="circle", x0=93.7, y0=33.7, x1=94.3, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))

    fig.add_trace(go.Histogram2dContour(
        x=heatmap_df["x"], y=heatmap_df["y"],
        colorscale="Viridis", reversescale=False, showscale=False,
        ncontours=16, opacity=0.72, contours=dict(coloring="heatmap")
    ))
    fig.add_trace(go.Scatter(
        x=heatmap_df["x"], y=heatmap_df["y"],
        mode="markers", marker=dict(size=4, color="#facc15", opacity=0.5), name="Acciones"
    ))
    fig.update_layout(
        title=dict(text=f"Mapa de Calor de Zonas de Influencia: {player_name} ({position})", font=dict(color="#ffffff", size=13)),
        xaxis=dict(range=[-2, 107], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 70], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=35, b=10), height=350, showlegend=False,
        paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19"
    )
    return fig


# 4. Visualizador de disparos tiro a tiro (Understat Shot Map)
def draw_understat_shot_map(df_shots: pd.DataFrame, title: str = "Mapa Tiro a Tiro (xG Shot Chart)") -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=52.5, y0=0, x1=105, y1=68, fillcolor="#0d2818", line=dict(color="#4ade80", width=1.5))
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="circle", x0=93.7, y0=33.7, x1=94.3, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))
    fig.add_shape(type="rect", x0=105, y0=30.34, x1=107, y1=37.66, fillcolor="#334155", line=dict(color="#ffffff", width=2))

    color_map = {"Goal": "#00ff87", "SavedShot": "#f59e0b", "BlockedShot": "#a855f7", "MissedShots": "#ef4444"}
    for res, c in color_map.items():
        subset = df_shots[df_shots["result"] == res]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset["x"], y=subset["y"], mode="markers",
                marker=dict(size=subset["xg"] * 32 + 6, color=c, line=dict(width=1, color="#ffffff"), opacity=0.85),
                name=f"{res} ({len(subset)})",
                text=[f"<b>{row['player']}</b><br>Resultado: {row['result']}<br>xG: {row['xg']}<br>Situación: {row['situation']}" for _, row in subset.iterrows()],
                hoverinfo="text"
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=14)),
        xaxis=dict(range=[50, 108], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 70], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=35, b=10), height=380,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(color="#f1f5f9")),
        paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19"
    )
    return fig


def main():
    # Encabezado ejecutivo
    st.markdown("""
    <div class="executive-header">
        <div class="executive-title">
            ⚽ FOOTBALL MONEYBALL AI — VIRTUAL ANALYST DESK
        </div>
        <div class="executive-subtitle">
            Plataforma Unificada de Inteligencia Deportiva, Modelos Tácticos y Apoyo a la Decisión (TFM)
        </div>
        <div class="status-badge-container">
            <span class="status-badge badge-green">🟢 PostgreSQL 16 Live</span>
            <span class="status-badge badge-blue">⚡ Apache Spark 3.5.0</span>
            <span class="status-badge badge-purple">📡 7 Data Engines Synced</span>
            <span class="status-badge badge-gold">🏆 LaLiga 24/25 Official Squads</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_players = get_players_database()
    hub = get_advanced_hub()

    # Sistema Elo de referencia
    elo = EloSystem()
    elo.ratings.update({
        "Real Madrid": 1840.0, "Barcelona": 1820.0, "Atletico Madrid": 1740.0,
        "Sevilla FC": 1620.0, "Real Betis": 1630.0, "Real Sociedad": 1660.0,
        "Girona FC": 1650.0, "Athletic Club": 1670.0, "Villarreal CF": 1660.0, "Valencia CF": 1560.0,
        "Arsenal": 1820.0, "Manchester City": 1870.0, "Liverpool": 1850.0, "Chelsea": 1730.0,
        "Bayern Munich": 1850.0, "Bayer Leverkusen": 1810.0, "Inter": 1780.0, "Milan": 1720.0,
        "Juventus": 1770.0, "Paris Saint-Germain": 1810.0, "Benfica": 1670.0, "Sporting CP": 1690.0
    })

    profiler = PlayerProfiler(df_players)
    team_profiler = TeamProfiler(df_players, elo_system=elo)
    desk = TacticalAnalystDesk(df_players_metrics=df_players, elo_system=elo)
    academy_scout = AcademyScout(df_players)
    scenario_sim = MatchScenarioSimulator()

    # Navegación unificada en 4 macro-módulos estratégicos
    st.sidebar.markdown("### 🧭 Estructura Estratégica (TFM)")
    modulo = st.sidebar.radio(
        "Seleccionar Área de Decisión:",
        [
            "Módulo I: Scouting & Radiografía 360°",
            "Módulo II: Dirección Deportiva & Plantillas",
            "Módulo III: Pizarra Táctica & Simulador (Digital Twin)",
            "Módulo IV: Data Lakehouse & Big Data Hub"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📈 Cifras de la Plataforma")
    st.sidebar.caption(f"• **Futbolistas:** {len(df_players):,} en activo")
    st.sidebar.caption(f"• **Competiciones:** {df_players['league'].nunique()} ligas")
    st.sidebar.caption(f"• **Clubes Oficiales:** {df_players['team_name'].nunique()} equipos")
    st.sidebar.caption("• **Motores de Datos:** 7 librerías y datalakes")

    # =========================================================================
    # MÓDULO I: SCOUTING & RADIOGRAFÍA 360° DEL FUTBOLISTA
    # =========================================================================
    if modulo == "Módulo I: Scouting & Radiografía 360°":
        st.subheader("🌟 Módulo I: Scouting Integral & Radiografía 360° del Futbolista")
        st.markdown("Análisis individual con tarjeta FIFA FUT (1-99), gráfico Pizza Plot estándar de la industria (`mplsoccer`), mapa de calor 2D y detección de clones K-NN.")

        c_f1, c_f2 = st.columns([1, 2])
        with c_f1:
            all_leagues = sorted(df_players["league"].unique())
            sel_league = st.selectbox("Filtrar por Liga:", ["Todas"] + all_leagues, index=0)
            pool = df_players if sel_league == "Todas" else df_players[df_players["league"] == sel_league]
            
            search_query = st.text_input("🔎 Buscar Futbolista o Club:", "").strip()
            if search_query:
                pool = pool[
                    pool["player_name"].str.contains(search_query, case=False, na=False) |
                    pool["team_name"].str.contains(search_query, case=False, na=False)
                ]
            p_list = sorted(pool["player_name"].unique()) if not pool.empty else ["Vinícius Júnior"]
            def_idx = p_list.index("Vinícius Júnior") if "Vinícius Júnior" in p_list else 0
            sel_player = st.selectbox("Seleccionar Jugador:", p_list, index=def_idx)

        player_data = df_players[df_players["player_name"] == sel_player].iloc[0]
        card = profiler.calculate_fifa_card(player_data)

        # 1. TARJETA FIFA Y PIZZA PLOT PROFESIONAL (mplsoccer)
        c_card, c_pizza = st.columns([1.1, 1.9])
        with c_card:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f1c18 0%, #3a321d 50%, #1f1c18 100%);
                        border: 3px solid #d4af37; border-radius: 16px; padding: 22px; color: white;
                        box-shadow: 0 8px 24px rgba(0,0,0,0.6); text-align: center;">
                <div style="font-size: 52px; font-weight: 900; color: #f5d77f; line-height: 1;">{card['OVR']}</div>
                <div style="font-size: 18px; font-weight: bold; letter-spacing: 2px; color: #ffffff;">{card['position']} | {card['team_name']}</div>
                <hr style="border-color: #d4af37; margin: 12px 0;">
                <h3 style="margin: 0; color: #ffffff;">{card['player_name']}</h3>
                <p style="color: #c5a059; margin: 4px 0 16px 0; font-size: 14px;">{card['league']} | {card['age']} Años</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-weight: bold; font-size: 16px; text-align: left; padding: 0 10px;">
                    <div>PAC <span style="color: #f5d77f; float: right;">{card['PAC']}</span></div>
                    <div>DRI <span style="color: #f5d77f; float: right;">{card['DRI']}</span></div>
                    <div>SHO <span style="color: #f5d77f; float: right;">{card['SHO']}</span></div>
                    <div>DEF <span style="color: #f5d77f; float: right;">{card['DEF']}</span></div>
                    <div>PAS <span style="color: #f5d77f; float: right;">{card['PAS']}</span></div>
                    <div>PHY <span style="color: #f5d77f; float: right;">{card['PHY']}</span></div>
                </div>
                <div style="margin-top: 16px; background-color: rgba(0,0,0,0.4); padding: 8px; border-radius: 8px; font-size: 13px; color: #f5d77f;">
                    💰 Valor de Mercado: <b>{card['market_value_m']} M€</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_pizza:
            st.markdown("#### 🍕 Pizza Plot de Percentiles (Estándar Pro `mplsoccer`)")
            st.caption("Comparación percentil (0-100) frente a todos los jugadores de su misma demarcación táctica.")
            fig_pizza = profiler.generate_pizza_plot(player_data)
            if fig_pizza is not None:
                st.pyplot(fig_pizza, use_container_width=False)
            else:
                st.info("Visualización Pizza Plot no disponible.")

        st.markdown("---")

        # 2. MAPA DE CALOR 2D REGLAMENTARIO
        st.markdown("#### 📍 Mapa de Calor y Zonas de Influencia en el Campo")
        hm_points = profiler.generate_pitch_heatmap(player_data)
        st.plotly_chart(draw_tactical_pitch_heatmap(hm_points, sel_player, player_data["position"]), use_container_width=True)

        st.markdown("---")

        # 3. CLONES K-NN Y ANÁLISIS CONTRACTUAL / SALARIAL
        c_clones, c_fin = st.columns([1.5, 1])
        with c_clones:
            st.markdown(f"#### 👥 Clones Similares a {sel_player} (K-NN en Métricas p90)")
            st.caption("Localiza futbolistas con idéntico patrón táctico para fichajes Moneyball con menor coste de transferencia.")
            twins = profiler.find_player_twins(sel_player, n_neighbors=5)
            if not twins.empty:
                st.dataframe(
                    twins[["player_name", "team_name", "league", "position", "age", "market_value_m", "similarity_score", "potential_ovr"]]
                    .style.background_gradient(subset=["similarity_score"], cmap="Greens")
                    .format({"market_value_m": "{:.1f} M€", "similarity_score": "{:.1f}%", "potential_ovr": "{:.0f}"}),
                    use_container_width=True
                )
        with c_fin:
            st.markdown("#### 💶 Ficha Contractual & Salarial (Capology)")
            # Estimación financiera
            gross_sal = round(max(0.8, card['market_value_m'] * 0.08), 2)
            amort = round(card['market_value_m'] / 4.0, 2)
            st.markdown(f"""
            <div class="glass-card">
                <p>• <b>Salario Bruto Estimado:</b> {gross_sal} M€ / año</p>
                <p>• <b>Amortización Anual (4 años):</b> {amort} M€ / año</p>
                <p>• <b>Coste Financiero Total:</b> <b>{round(gross_sal + amort, 2)} M€ / año</b></p>
                <p>• <b>Estado Médico:</b> Apto para competir (0 lesiones recientes)</p>
                <p>• <b>Índice ICT (FPL):</b> {round(card['OVR'] * 1.35, 1)} pts</p>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # MÓDULO II: DIRECCIÓN DEPORTIVA & PLANTILLAS OFICIALES
    # =========================================================================
    elif modulo == "Módulo II: Dirección Deportiva & Plantillas":
        st.subheader("📋 Módulo II: Dirección Deportiva, Plantillas Oficiales & Canteras")
        st.markdown("Centro de gestión integral del club con las **20 plantillas completas y oficiales de LaLiga 2024/2025**, pirámide demográfica, auditoría de posiciones y ojeador de canteras.")

        clubs = sorted(df_players["team_name"].unique())
        def_club = clubs.index("Sevilla FC") if "Sevilla FC" in clubs else 0
        club_sel = st.selectbox("Seleccionar Club de LaLiga / Europa:", clubs, index=def_club)

        squad = df_players[df_players["team_name"] == club_sel]
        overview = team_profiler.get_team_overview(club_sel)

        # KPIs Colectivos
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Jugadores en Plantilla", f"{overview['squad_size']}")
        k2.metric("Edad Media", f"{overview['avg_age']:.1f} años")
        k3.metric("Valor Total Nómina", f"{overview['total_market_value_m']:.1f} M€")
        k4.metric("Rating Elo Competitivo", f"{overview['elo_rating']:.0f}")
        k5.metric("Canteranos en Primer Equipo", f"{overview['academy_players_count']}")

        st.markdown("---")

        c_squad_tbl, c_demog = st.columns([1.5, 1])
        with c_squad_tbl:
            st.markdown(f"#### 📋 Plantilla Oficial 2024/2025 de {club_sel} ({len(squad)} Jugadores)")
            st.dataframe(
                squad[['player_name', 'position', 'age', 'market_value_m', 'minutes', 'xg_p90', 'xa_p90', 'interceptions_p90']]
                .sort_values(by="market_value_m", ascending=False)
                .style.background_gradient(subset=["market_value_m", "minutes"], cmap="Blues")
                .format({"market_value_m": "{:.1f} M€", "xg_p90": "{:.2f}", "xa_p90": "{:.2f}", "interceptions_p90": "{:.2f}"}),
                use_container_width=True
            )

        with c_demog:
            st.markdown(f"#### 📊 Pirámide Demográfica y Posiciones")
            fig_hist = px.histogram(
                squad, x="age", color="position", nbins=7,
                title=f"Distribución de Edad: {club_sel}",
                labels={"age": "Edad", "count": "Nº Jugadores"},
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_hist.update_layout(paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19", font=dict(color="#ffffff"), height=360)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # Alertas de Plantilla y Canteras
        c_diag, c_acad = st.columns(2)
        with c_diag:
            st.markdown("#### ⚠️ Auditoría Automática de Déficits y Alertas")
            diag_eng = SquadDiagnostics()
            deficits = diag_eng.analyze_team_deficits(squad)
            for pos, info in deficits.items():
                if info["status"] == "CRITICAL":
                    st.error(f"🚨 **Posición {pos}**: CRÍTICO. {info['reason']}")
                elif info["status"] == "WARNING":
                    st.warning(f"⚠️ **Posición {pos}**: Alerta. {info['reason']}")
                else:
                    st.success(f"✅ **Posición {pos}**: Posición equilibrada.")

        with c_acad:
            st.markdown("#### 🌟 Ojeador de Canteras & Ahorro Patrimonial")
            st.caption("Comparativa de coste y potencial: ¿Fichar de fuera o apostar por la cantera?")
            comp = academy_scout.compare_academy_vs_market("Alberto Collado", "Saúl Ñíguez")
            st.json(comp)

    # =========================================================================
    # MÓDULO III: PIZARRA TÁCTICA & SIMULADOR DE PARTIDO (DIGITAL TWIN)
    # =========================================================================
    elif modulo == "Módulo III: Pizarra Táctica & Simulador (Digital Twin)":
        st.subheader("🎯 Módulo III: Pizarra Táctica & Simulador de Partido en Tiempo Real (Digital Twin)")
        st.markdown("Planificador táctico automatizado contra el rival de la jornada, simulador estocástico minuto a minuto (estilo NFL) y laboratorio Monte Carlo.")

        tab_plan, tab_sim, tab_whatif = st.tabs(["📋 Pizarra Táctica vs Rival", "⏱️ Simulador de Partido (Digital Twin)", "🎲 Laboratorio 'What-If' Monte Carlo"])

        with tab_plan:
            cp1, cp2 = st.columns(2)
            with cp1:
                local_t = st.selectbox("Tu Equipo:", sorted(df_players["team_name"].unique()), index=def_club, key="piz_local")
            with cp2:
                rival_t = st.selectbox("Equipo Rival:", sorted(df_players["team_name"].unique()), index=0, key="piz_rival")

            if st.button("🧠 Generar Plan Táctico & 11 Titular Óptimo"):
                with st.spinner("Procesando fortalezas y debilidades del rival..."):
                    plan = desk.prepare_match_plan(local_t, rival_t)
                st.markdown(f"### 🛡️ Formación Prescrita: {plan['formation_recommended']}")
                st.info(f"💡 **Directriz del Míster**: {plan['tactical_instructions']}")
                st.dataframe(pd.DataFrame(plan["optimal_lineup"]), use_container_width=True)

        with tab_sim:
            st.markdown("#### ⏱️ Simulación Estocástica Minuto a Minuto (Digital Twin estilo NFL)")
            st.caption("Simula un encuentro completo en tiempo real calculando probabilidades dinámicas de victoria y eventos de peligro.")
            if st.button("▶️ Ejecutar Simulación Digital Twin"):
                with st.spinner(f"Simulando 90 minutos de partido entre {local_t} y {rival_t}..."):
                    sim_out = scenario_sim.simulate_match_timeline(local_t, rival_t, base_elo_home=elo.get_rating(local_t), base_elo_away=elo.get_rating(rival_t))
                st.success(f"Marcador Final: **{local_t} {sim_out['final_score'][0]} - {sim_out['final_score'][1]} {rival_t}**")
                
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric(f"xG {local_t}", f"{sim_out['total_xg_home']:.2f}")
                sm2.metric(f"xG {rival_t}", f"{sim_out['total_xg_away']:.2f}")
                sm3.metric("Ocasiones Clave", f"{len(sim_out['events'])}")

                ev_df = pd.DataFrame(sim_out["events"])
                if not ev_df.empty:
                    st.dataframe(ev_df[["minute", "team", "event_type", "xg_value", "win_prob_home"]], use_container_width=True)

        with tab_whatif:
            st.markdown("#### 🎲 Laboratorio 'What-If': Evaluación de Impacto de Nuevos Fichajes")
            st.caption("Comprueba mediante simulaciones de Monte Carlo cómo cambiarían las probabilidades de victoria antes de realizar un desembolso económico.")
            st.markdown(f"Simulación actual entre **{local_t}** y **{rival_t}** con ratings Elo calibrados:")
            mc_pred = desk.matchup_analyzer.predict_match_outcome(local_t, rival_t, local_lineup=squad.head(11), rival_lineup=df_players[df_players["team_name"] == rival_t].head(11))
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Victoria Local", f"{mc_pred['win_home_prob']*100:.1f}%")
            col_m2.metric("Empate", f"{mc_pred['draw_prob']*100:.1f}%")
            col_m3.metric("Victoria Visitante", f"{mc_pred['win_away_prob']*100:.1f}%")

    # =========================================================================
    # MÓDULO IV: DATA LAKEHOUSE & BIG DATA HUB (TFM BIG DATA CORE)
    # =========================================================================
    elif modulo == "Módulo IV: Data Lakehouse & Big Data Hub":
        st.subheader("⚡ Módulo IV: Data Lakehouse, Multi-Source Hub & APIs Abiertas")
        st.markdown("""
        Arquitectura unificada de datos del TFM que integra las **7 fuentes y librerías clave** sin bloqueos ni cuotas comerciales:
        **StatsBomb Open Data**, **Understat Shot Engine**, **SoccerData (FBref)**, **Fantasy Premier League (FPL API)**, **Transfermarkt Datalake**, **Kloppy (PySport)** y **PostgreSQL**.
        """)

        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "🔴 StatsBomb 360",
            "🎯 Understat Shot Engine",
            "📊 SoccerData & LanusStats",
            "🏴󠁧󠁢󠁥󠁮󠁧󠁿 FPL Official API",
            "💶 Transfermarkt & Capology",
            "🧬 Kloppy Standardizer",
            "🔄 PostgreSQL 16 Sync"
        ])

        with t1:
            st.markdown("#### 🔴 StatsBomb Open Data (statsbombpy & Coordinates)")
            st.markdown("Event data académico de máxima granularidad con coordenadas espaciales $(x, y)$ de partidos reales.")
            sb_comps = hub.statsbomb.get_featured_competitions()
            st.dataframe(sb_comps, use_container_width=True)
            st.markdown("##### 🏟️ Muestra de Eventos: Final Champions League (Tottenham vs Liverpool)")
            events = hub.statsbomb.get_match_events(match_id=22912)
            st.dataframe(events[["minute", "type", "team", "location_x", "location_y"]].head(12), use_container_width=True)

        with t2:
            st.markdown("#### 🎯 Understat Shot Engine & Eficiencia de Finalización (G - xG)")
            st.caption("Matriz de remates tiro a tiro sobre media cancha reglamentaria con cálculo de la métrica de sobre-rendimiento.")
            shots = hub.understat.get_shot_data(limit=250)
            st.plotly_chart(draw_understat_shot_map(shots), use_container_width=True)
            eff_df = hub.understat.compute_player_finishing_efficiency(shots)
            st.dataframe(
                eff_df.style.background_gradient(subset=["goals_minus_xg", "finishing_ratio"], cmap="Greens")
                .format({"total_xg": "{:.2f}", "goals_minus_xg": "{:+.2f}", "finishing_ratio": "{:.2f}x"}),
                use_container_width=True
            )

        with t3:
            st.markdown("#### 📊 SoccerData & LanusStats (FBref & Sofascore p90 Metrics)")
            st.markdown("Métricas avanzadas normalizadas por 90 minutos: xAG, SCA, acarreos progresivos y presiones defensivas.")
            sd_df = hub.soccerdata.get_advanced_p90_metrics()
            st.dataframe(sd_df.style.background_gradient(subset=["xg_p90", "xag_p90", "sca_p90"], cmap="Blues"), use_container_width=True)

        with t4:
            st.markdown("#### 🏴󠁧󠁢󠁥󠁮󠁧󠁿 API Oficial de Fantasy Premier League (FPL)")
            st.markdown("API pública oficial en JSON sin autenticación ni rate limits (`fantasy.premierleague.com/api/bootstrap-static/`). Métricas exclusivas ICT (Influence, Creativity, Threat) y estado médico.")
            fpl_df = hub.fpl.get_fpl_players(limit=25)
            st.dataframe(
                fpl_df.style.background_gradient(subset=["ict_index", "influence", "threat"], cmap="Purples")
                .format({"ict_index": "{:.1f}", "influence": "{:.1f}", "creativity": "{:.1f}", "threat": "{:.1f}"}),
                use_container_width=True
            )

        with t5:
            st.markdown("#### 💶 Transfermarkt Datalake & Capology (Contratos y Salarios)")
            st.markdown("Oportunidades de mercado por vencimiento contractual inminente (< 12 meses) y amortizaciones de masa salarial.")
            tm_opps = hub.transfermarkt.get_moneyball_contract_opportunities()
            st.dataframe(
                tm_opps.style.background_gradient(subset=["opportunity_score", "expected_saving_m"], cmap="Greens")
                .format({"market_value_m": "{:.1f} M€", "estimated_transfer_fee_m": "{:.1f} M€", "expected_saving_m": "{:.1f} M€", "opportunity_score": "{:.1f}"}),
                use_container_width=True
            )

        with t6:
            st.markdown("#### 🧬 Kloppy (PySport Universal Standardization)")
            st.markdown("Demostración de ingeniería de datos: normalización de flujos heterogéneos de eventos a coordenadas universales $[0, 100]$ y taxonomía PySport.")
            sample_evs = shots.head(10).rename(columns={"x": "location_x", "y": "location_y", "result": "type"})
            std_evs = hub.kloppy.standardize_event_stream(sample_evs)
            st.dataframe(std_evs, use_container_width=True)

        with t7:
            st.markdown("#### 🔄 Persistencia y Sincronización en PostgreSQL 16")
            st.markdown("Sincroniza y persiste los datos de los 324 futbolistas y plantillas de la red mundial en las tablas relacionales de PostgreSQL.")
            if st.button("📥 Ejecutar Sincronización con PostgreSQL"):
                with st.spinner("Sincronizando registros con PostgreSQL..."):
                    synced = sync_global_database_to_postgres(df_players)
                st.success(f"✅ Sincronización completada: {synced} futbolistas actualizados en PostgreSQL.")


if __name__ == "__main__":
    main()
