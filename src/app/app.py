import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Asegurar raíz del proyecto en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

# Importaciones de la arquitectura analítica unificada del TFM
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


# =============================================================================
# 1. CONFIGURACIÓN Y ESTILOS DE PÁGINA
# =============================================================================
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
    .fifa-card-fut {
        background: linear-gradient(135deg, #1f1c18 0%, #3a321d 50%, #1f1c18 100%);
        border: 3px solid #d4af37;
        border-radius: 16px;
        padding: 22px;
        color: white;
        box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. CONECTORES CACHEADOS Y DEFENSA ANTE FALLOS
# =============================================================================
@st.cache_resource
def get_db_engine():
    try:
        return create_engine("postgresql://moneyball:scouting_admin@postgres:5432/football_db")
    except Exception:
        try:
            return create_engine("postgresql://moneyball:scouting_admin@localhost:5432/football_db")
        except Exception:
            return None

@st.cache_data
def get_players_database(season: str = "2026/2027") -> pd.DataFrame:
    """Carga la base de datos de futbolistas reales con sus métricas y valores oficiales para la temporada solicitada."""
    try:
        df = load_global_players(season=season)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        st.warning(f"Aviso al cargar base de datos global ({season}): {e}. Activando generador local de contingencia.")
    from src.data.global_player_database import generate_worldwide_player_network
    return generate_worldwide_player_network(season=season)

@st.cache_resource
def get_advanced_hub() -> MasterAnalyticsDataHub:
    try:
        return MasterAnalyticsDataHub()
    except Exception:
        return MasterAnalyticsDataHub()


# =============================================================================
# 3. VISUALIZADORES TÁCTICOS 2D
# =============================================================================
def draw_tactical_pitch_heatmap(heatmap_df: pd.DataFrame, player_name: str, position: str) -> go.Figure:
    """Genera el gráfico interactivo de campo 105x68 con el mapa de calor 2D del futbolista."""
    fig = go.Figure()
    # Límites del campo reglamentario
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, fillcolor="#0d2818", line=dict(color="#4ade80", width=1.5))
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="circle", x0=52.5 - 9.15, y0=34 - 9.15, x1=52.5 + 9.15, y1=34 + 9.15, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="circle", x0=52.2, y0=33.7, x1=52.8, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))
    # Áreas
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="circle", x0=10.7, y0=33.7, x1=11.3, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))
    fig.add_shape(type="circle", x0=93.7, y0=33.7, x1=94.3, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))

    if heatmap_df is not None and not heatmap_df.empty and "x" in heatmap_df.columns and "y" in heatmap_df.columns:
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


def draw_understat_shot_map(df_shots: pd.DataFrame, title: str = "Mapa Tiro a Tiro (xG Shot Chart)") -> go.Figure:
    """Genera el visualizador de disparos tiro a tiro sobre media cancha reglamentaria."""
    fig = go.Figure()
    fig.add_shape(type="rect", x0=52.5, y0=0, x1=105, y1=68, fillcolor="#0d2818", line=dict(color="#4ade80", width=1.5))
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="circle", x0=93.7, y0=33.7, x1=94.3, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))
    fig.add_shape(type="rect", x0=105, y0=30.34, x1=107, y1=37.66, fillcolor="#334155", line=dict(color="#ffffff", width=2))

    color_map = {"Goal": "#00ff87", "SavedShot": "#f59e0b", "BlockedShot": "#a855f7", "MissedShots": "#ef4444"}
    if df_shots is not None and not df_shots.empty:
        for res, c in color_map.items():
            subset = df_shots[df_shots.get("result", "") == res]
            if not subset.empty:
                hover_txt = [
                    f"<b>{row.get('player', 'Jugador')}</b><br>Resultado: {row.get('result', '')}<br>xG: {row.get('xg', 0.0):.2f}<br>Situación: {row.get('situation', 'Juego Abierto')}"
                    for _, row in subset.iterrows()
                ]
                fig.add_trace(go.Scatter(
                    x=subset["x"], y=subset["y"], mode="markers",
                    marker=dict(size=np.clip(subset["xg"] * 32 + 6, 6, 35), color=c, line=dict(width=1, color="#ffffff"), opacity=0.85),
                    name=f"{res} ({len(subset)})",
                    text=hover_txt,
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


def draw_digital_twin_xg_timeline(timeline_df: pd.DataFrame, home_team: str, away_team: str) -> go.Figure:
    """Gráfico interactivo de evolución de xG acumulado durante los 90 minutos de partido."""
    fig = go.Figure()
    if timeline_df is not None and not timeline_df.empty:
        fig.add_trace(go.Scatter(
            x=timeline_df["minute"], y=timeline_df["xg_home"],
            mode="lines", name=f"xG Acumulado {home_team}",
            line=dict(color="#00ff87", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=timeline_df["minute"], y=timeline_df["xg_away"],
            mode="lines", name=f"xG Acumulado {away_team}",
            line=dict(color="#38bdf8", width=3, dash="dot")
        ))
    fig.update_layout(
        title=dict(text=f"Evolución Estocástica de Goles Esperados (xG) 90 Minutos: {home_team} vs {away_team}", font=dict(color="#ffffff", size=13)),
        xaxis=dict(title="Minuto de Juego", range=[0, 92], showgrid=True, gridcolor="#1e293b", color="#94a3b8"),
        yaxis=dict(title="xG Acumulado", showgrid=True, gridcolor="#1e293b", color="#94a3b8"),
        margin=dict(l=20, r=20, t=40, b=20), height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color="#f1f5f9")),
        paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19"
    )
    return fig


# =============================================================================
# 4. APLICACIÓN PRINCIPAL STREAMLIT
# =============================================================================
def main():
    # Barra lateral: Selector de temporada oficial (2026/2027 en curso por defecto)
    st.sidebar.markdown("### 📅 Temporada de Análisis")
    season_options = [
        "2026/2027 (En Curso)",
        "2025/2026",
        "2024/2025"
    ]
    season_sel_str = st.sidebar.selectbox(
        "Seleccionar Temporada Oficial:",
        season_options,
        index=0,
        key="global_season_selector"
    )
    clean_season = season_sel_str.split(" ")[0]

    # Encabezado ejecutivo de la plataforma
    st.markdown(f"""
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
            <span class="status-badge badge-gold">🏆 Temporada {clean_season} Activa</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Carga defensiva de la base de datos de jugadores según la temporada activa
    try:
        df_players = get_players_database(season=clean_season)
    except Exception as e:
        st.error(f"Error crítico al conectar con la base de datos de jugadores ({clean_season}): {e}")
        df_players = pd.DataFrame()

    if df_players.empty:
        st.error("No se han podido cargar futbolistas. Por favor, reinicie la base de datos.")
        return

    hub = get_advanced_hub()

    # Calibración del Sistema Elo
    elo = EloSystem()
    elo.ratings.update({
        "Real Madrid": 1840.0, "Barcelona": 1820.0, "Atletico Madrid": 1740.0,
        "Sevilla FC": 1620.0, "Real Betis": 1630.0, "Real Sociedad": 1660.0,
        "Girona FC": 1650.0, "Athletic Club": 1670.0, "Villarreal CF": 1660.0, "Valencia CF": 1560.0,
        "Arsenal": 1820.0, "Manchester City": 1870.0, "Liverpool": 1850.0, "Chelsea": 1730.0,
        "Bayern Munich": 1850.0, "Bayer Leverkusen": 1810.0, "Inter": 1780.0, "Milan": 1720.0,
        "Juventus": 1770.0, "Paris Saint-Germain": 1810.0, "Benfica": 1670.0, "Sporting CP": 1690.0
    })

    # Motores analíticos
    profiler = PlayerProfiler(df_players)
    team_profiler = TeamProfiler(df_players, elo_system=elo)
    desk = TacticalAnalystDesk(df_players_metrics=df_players, elo_system=elo)
    academy_scout = AcademyScout(df_players)
    scenario_sim = MatchScenarioSimulator()

    # Barra lateral de navegación
    st.sidebar.markdown("### 🧭 Estructura Estratégica (TFM)")
    modulo = st.sidebar.radio(
        "Seleccionar Área de Decisión:",
        [
            "Módulo I: Scouting & Radiografía 360°",
            "Módulo II: Dirección Deportiva & Plantillas",
            "Módulo III: Pizarra Táctica & Simulador (Digital Twin)",
            "Módulo IV: Data Lakehouse & Big Data Hub"
        ],
        key="main_nav_selector"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📈 Cifras de la Plataforma")
    n_players = len(df_players)
    n_leagues = df_players["league"].nunique() if "league" in df_players.columns else 0
    n_teams = df_players["team_name"].nunique() if "team_name" in df_players.columns else 0
    st.sidebar.caption(f"• **Futbolistas:** {n_players:,} en activo")
    st.sidebar.caption(f"• **Competiciones:** {n_leagues} ligas de élite")
    st.sidebar.caption(f"• **Clubes Oficiales:** {n_teams} equipos censados")
    st.sidebar.caption("• **Motores de Datos:** 7 librerías y datalakes")

    # Lista global de clubes disponibles
    available_clubs = sorted(df_players["team_name"].dropna().unique()) if "team_name" in df_players.columns else ["Sevilla FC"]
    def_club_idx = available_clubs.index("Sevilla FC") if "Sevilla FC" in available_clubs else 0

    # =========================================================================
    # MÓDULO I: SCOUTING & RADIOGRAFÍA 360° DEL FUTBOLISTA
    # =========================================================================
    if modulo == "Módulo I: Scouting & Radiografía 360°":
        st.subheader("🌟 Módulo I: Scouting Integral & Radiografía 360° del Futbolista")
        st.markdown("Análisis individual con tarjeta FIFA FUT (1-99), gráfico Pizza Plot estándar de la industria (`mplsoccer`), mapa de calor 2D y detección de clones K-NN.")

        try:
            c_f1, c_f2 = st.columns([1, 2])
            with c_f1:
                all_leagues = sorted(df_players["league"].dropna().unique()) if "league" in df_players.columns else []
                sel_league = st.selectbox("Filtrar por Liga:", ["Todas"] + all_leagues, index=0, key="mod1_league_filter")
                pool = df_players if sel_league == "Todas" else df_players[df_players["league"] == sel_league]
                
                search_query = st.text_input("🔎 Buscar Futbolista o Club:", "", key="mod1_search_input").strip()
                if search_query:
                    filtered_pool = pool[
                        pool["player_name"].str.contains(search_query, case=False, na=False) |
                        pool["team_name"].str.contains(search_query, case=False, na=False)
                    ]
                    if not filtered_pool.empty:
                        pool = filtered_pool
                    else:
                        st.warning("⚠️ No se encontraron futbolistas con el criterio introducido. Mostrando jugadores disponibles.")

                p_list = sorted(pool["player_name"].dropna().unique()) if not pool.empty else sorted(df_players["player_name"].dropna().unique())
                def_idx = p_list.index("Vinícius Júnior") if "Vinícius Júnior" in p_list else 0
                sel_player = st.selectbox("Seleccionar Jugador:", p_list, index=def_idx, key="mod1_player_selector")

            player_matches = df_players[df_players["player_name"] == sel_player]
            player_data = player_matches.iloc[0] if not player_matches.empty else df_players.iloc[0]
            card = profiler.calculate_fifa_card(player_data)
        except Exception as e:
            st.error(f"Error procesando el perfil del jugador: {e}")
            return

        # 1. TARJETA FIFA Y PIZZA PLOT PROFESIONAL (mplsoccer)
        c_card, c_pizza = st.columns([1.1, 1.9])
        with c_card:
            try:
                st.markdown(f"""
                <div class="fifa-card-fut">
                    <div style="font-size: 52px; font-weight: 900; color: #f5d77f; line-height: 1;">{card.get('OVR', 85)}</div>
                    <div style="font-size: 18px; font-weight: bold; letter-spacing: 2px; color: #ffffff;">{card.get('position', 'MF')} | {card.get('team_name', 'Club')}</div>
                    <hr style="border-color: #d4af37; margin: 12px 0;">
                    <h3 style="margin: 0; color: #ffffff;">{card.get('player_name', sel_player)}</h3>
                    <p style="color: #c5a059; margin: 4px 0 16px 0; font-size: 14px;">{card.get('league', 'Liga')} | {card.get('age', 25)} Años</p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-weight: bold; font-size: 16px; text-align: left; padding: 0 10px;">
                        <div>PAC <span style="color: #f5d77f; float: right;">{card.get('PAC', 75)}</span></div>
                        <div>DRI <span style="color: #f5d77f; float: right;">{card.get('DRI', 75)}</span></div>
                        <div>SHO <span style="color: #f5d77f; float: right;">{card.get('SHO', 75)}</span></div>
                        <div>DEF <span style="color: #f5d77f; float: right;">{card.get('DEF', 75)}</span></div>
                        <div>PAS <span style="color: #f5d77f; float: right;">{card.get('PAS', 75)}</span></div>
                        <div>PHY <span style="color: #f5d77f; float: right;">{card.get('PHY', 75)}</span></div>
                    </div>
                    <div style="margin-top: 16px; background-color: rgba(0,0,0,0.4); padding: 8px; border-radius: 8px; font-size: 13px; color: #f5d77f;">
                        💰 Valor de Mercado: <b>{card.get('market_value_m', 10.0):.1f} M€</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generando la tarjeta FIFA: {e}")

        with c_pizza:
            st.markdown("#### 🍕 Pizza Plot de Percentiles (Estándar Pro `mplsoccer`)")
            st.caption("Comparación percentil (0-100) frente a todos los jugadores de su misma demarcación táctica.")
            try:
                import matplotlib.pyplot as plt
                fig_pizza = profiler.generate_pizza_plot(player_data)
                if fig_pizza is not None:
                    st.pyplot(fig_pizza, use_container_width=False)
                    plt.close(fig_pizza)
                else:
                    st.info("Visualización Pizza Plot no disponible en el entorno actual.")
            except Exception as e:
                st.warning(f"Aviso en Pizza Plot: {e}")

        st.markdown("---")

        # 2. MAPA DE CALOR 2D REGLAMENTARIO
        st.markdown("#### 📍 Mapa de Calor y Zonas de Influencia en el Campo")
        try:
            hm_points = profiler.get_pitch_heatmap_data(player_data)
            st.plotly_chart(draw_tactical_pitch_heatmap(hm_points, sel_player, player_data.get("position", "MF")), use_container_width=True)
        except Exception as e:
            st.warning(f"Aviso generando mapa de calor: {e}")

        st.markdown("---")

        # 3. CLONES K-NN Y ANÁLISIS CONTRACTUAL / SALARIAL
        c_clones, c_fin = st.columns([1.5, 1])
        with c_clones:
            st.markdown(f"#### 👥 Clones Similares a {sel_player} (K-NN en Métricas p90)")
            st.caption("Localiza futbolistas con idéntico patrón táctico para fichajes Moneyball con menor coste de transferencia.")
            try:
                twins = profiler.find_player_twins(player_name=sel_player, top_k=5)
                if twins is not None and not twins.empty:
                    cols_to_show = [c for c in ["player_name", "team_name", "league", "position", "age", "market_value_m", "similarity_pct", "potential_ovr"] if c in twins.columns]
                    fmt_dict = {}
                    if "market_value_m" in cols_to_show:
                        fmt_dict["market_value_m"] = "{:.1f} M€"
                    if "similarity_pct" in cols_to_show:
                        fmt_dict["similarity_pct"] = "{:.1f}%"
                    if "potential_ovr" in cols_to_show:
                        fmt_dict["potential_ovr"] = "{:.0f}"

                    st_df = twins[cols_to_show].style
                    if "similarity_pct" in cols_to_show:
                        st_df = st_df.background_gradient(subset=["similarity_pct"], cmap="Greens")
                    st.dataframe(st_df.format(fmt_dict), use_container_width=True)
                else:
                    st.info("No se encontraron clones directos para la posición seleccionada.")
            except Exception as e:
                st.warning(f"Aviso en cálculo de clones K-NN: {e}")

        with c_fin:
            st.markdown("#### 💶 Ficha Contractual & Salarial (Capology)")
            try:
                mv = float(card.get('market_value_m', 10.0))
                gross_sal = round(max(0.8, mv * 0.08), 2)
                amort = round(mv / 4.0, 2)
                tot_fin = round(gross_sal + amort, 2)
                ict_est = round(float(card.get('OVR', 80)) * 1.35, 1)
                st.markdown(f"""
                <div class="glass-card">
                    <p>• <b>Salario Bruto Estimado:</b> {gross_sal} M€ / año</p>
                    <p>• <b>Amortización Anual (4 años):</b> {amort} M€ / año</p>
                    <p>• <b>Coste Financiero Total:</b> <b>{tot_fin} M€ / año</b></p>
                    <p>• <b>Estado Médico:</b> Apto para competir (0 lesiones recientes)</p>
                    <p>• <b>Índice ICT (FPL):</b> {ict_est} pts</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Aviso en ficha financiera: {e}")

    # =========================================================================
    # MÓDULO II: DIRECCIÓN DEPORTIVA & PLANTILLAS OFICIALES
    # =========================================================================
    elif modulo == "Módulo II: Dirección Deportiva & Plantillas":
        st.subheader("📋 Módulo II: Dirección Deportiva, Plantillas Oficiales & Canteras")
        st.markdown(f"Centro de gestión integral del club con las **20 plantillas completas y oficiales de LaLiga {clean_season}**, pirámide demográfica, auditoría de posiciones y ojeador de canteras.")

        try:
            club_sel = st.selectbox(
                "Seleccionar Club de LaLiga / Europa:",
                available_clubs,
                index=def_club_idx,
                key="mod2_club_selector"
            )

            squad = df_players[df_players["team_name"] == club_sel]
            overview = team_profiler.get_team_overview(club_sel)
        except Exception as e:
            st.error(f"Error cargando los datos del club {club_sel}: {e}")
            return

        # KPIs Colectivos
        k1, k2, k3, k4, k5 = st.columns(5)
        squad_size = overview.get("squad_size", len(squad))
        avg_age = overview.get("avg_age", squad["age"].mean() if "age" in squad.columns else 25.0)
        tot_val = overview.get("total_market_value_m", squad["market_value_m"].sum() if "market_value_m" in squad.columns else 0.0)
        elo_val = overview.get("elo_rating", elo.get_rating(club_sel))
        acad_val = overview.get("academy_players_count", int(squad["is_academy"].fillna(False).sum()) if "is_academy" in squad.columns else 0)

        k1.metric("Jugadores en Plantilla", f"{squad_size}")
        k2.metric("Edad Media", f"{avg_age:.1f} años")
        k3.metric("Valor Total Nómina", f"{tot_val:.1f} M€")
        k4.metric("Rating Elo Competitivo", f"{elo_val:.0f}")
        k5.metric("Canteranos en Primer Equipo", f"{acad_val}")

        st.markdown("---")

        c_squad_tbl, c_demog = st.columns([1.5, 1])
        with c_squad_tbl:
            st.markdown(f"#### 📋 Plantilla Oficial {clean_season} de {club_sel} ({len(squad)} Jugadores)")
            try:
                target_cols = ['player_name', 'position', 'age', 'market_value_m', 'minutes', 'xg_p90', 'xa_p90', 'interceptions_p90']
                show_cols = [c for c in target_cols if c in squad.columns]
                squad_sorted = squad[show_cols].sort_values(by="market_value_m", ascending=False) if "market_value_m" in show_cols else squad[show_cols]
                
                fmt_squad = {}
                if "market_value_m" in show_cols: fmt_squad["market_value_m"] = "{:.1f} M€"
                if "xg_p90" in show_cols: fmt_squad["xg_p90"] = "{:.2f}"
                if "xa_p90" in show_cols: fmt_squad["xa_p90"] = "{:.2f}"
                if "interceptions_p90" in show_cols: fmt_squad["interceptions_p90"] = "{:.2f}"

                st_squad = squad_sorted.style
                grad_subset = [c for c in ["market_value_m", "minutes"] if c in show_cols]
                if grad_subset:
                    st_squad = st_squad.background_gradient(subset=grad_subset, cmap="Blues")
                st.dataframe(st_squad.format(fmt_squad), use_container_width=True)
            except Exception as e:
                st.warning(f"Aviso renderizando plantilla: {e}")

        with c_demog:
            st.markdown(f"#### 📊 Pirámide Demográfica y Posiciones")
            try:
                fig_hist = px.histogram(
                    squad, x="age", color="position", nbins=7,
                    title=f"Distribución de Edad: {club_sel}",
                    labels={"age": "Edad", "count": "Nº Jugadores"},
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_hist.update_layout(paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19", font=dict(color="#ffffff"), height=360)
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.warning(f"Aviso en pirámide demográfica: {e}")

        st.markdown("---")

        # Alertas de Plantilla y Canteras
        c_diag, c_acad = st.columns(2)
        with c_diag:
            st.markdown("#### ⚠️ Auditoría Automática de Déficits y Alertas")
            try:
                diag_eng = SquadDiagnostics(df_players)
                deficits = diag_eng.analyze_team_deficits(team_name=club_sel, metric="xg_p90")
                if deficits is not None and not deficits.empty:
                    for _, row in deficits.iterrows():
                        pos = row.get("position", "Línea")
                        status = row.get("status", "OK")
                        reason = row.get("reason", "")
                        if status == "CRITICAL":
                            st.error(f"🚨 **Posición {pos}**: CRÍTICO. {reason}")
                        elif status == "WARNING":
                            st.warning(f"⚠️ **Posición {pos}**: Alerta. {reason}")
                        else:
                            st.success(f"✅ **Posición {pos}**: Posición equilibrada ({reason}).")
                else:
                    st.info("No se han detectado déficits anómalos en el equipo.")
            except Exception as e:
                st.warning(f"Aviso en diagnóstico de plantilla: {e}")

        with c_acad:
            st.markdown("#### 🌟 Ojeador de Canteras & Ahorro Patrimonial")
            st.caption("Comparativa de coste y potencial: ¿Fichar de fuera o apostar por la cantera?")
            try:
                # Detección inteligente de canterano representativo del club y estrella de mercado
                squad_acad = squad[squad["is_academy"] == True] if "is_academy" in squad.columns else pd.DataFrame()
                canterano_def = squad_acad["player_name"].iloc[0] if not squad_acad.empty else "Alberto Collado"
                
                # Opciones interactivas
                acad_candidates = sorted(squad_acad["player_name"].unique()) if not squad_acad.empty else ["Alberto Collado", "Marc Casadó", "Lamine Yamal", "Juanlu Sánchez"]
                senior_candidates = ["Saúl Ñíguez", "Rodri", "Luka Modric", "İkay Gündoğan", "Frenkie de Jong", "Joshua Kimmich"]
                
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    sel_canterano = st.selectbox("Canterano del Club:", acad_candidates, index=0, key="mod2_sel_canterano")
                with col_c2:
                    sel_senior = st.selectbox("Fichaje Senior de Referencia:", senior_candidates, index=0, key="mod2_sel_senior")

                comp = academy_scout.compare_academy_vs_market_signing(sel_canterano, sel_senior)
                
                verdict = comp.get("financial_verdict", {})
                net_savings = verdict.get("net_savings_eur_m", 0.0)
                ffp_relief = verdict.get("ffp_relief_annual_m", 0.0)
                
                m_ac1, m_ac2 = st.columns(2)
                m_ac1.metric("Ahorro Neto Previsto (4-5 Años)", f"{net_savings:+.1f} M€")
                m_ac2.metric("Alivio Fair Play Anual", f"{ffp_relief:+.1f} M€/año")
                st.info(f"💡 **Dictamen del Director Deportivo**: {verdict.get('verdict_summary', '')}")
                
                with st.expander("🔍 Ver Desglose Financiero & Salarial Detallado"):
                    st.json(comp)
            except Exception as e:
                st.warning(f"Aviso en comparativa de cantera vs mercado: {e}")

    # =========================================================================
    # MÓDULO III: PIZARRA TÁCTICA & SIMULADOR DE PARTIDO (DIGITAL TWIN)
    # =========================================================================
    elif modulo == "Módulo III: Pizarra Táctica & Simulador (Digital Twin)":
        st.subheader("🎯 Módulo III: Pizarra Táctica & Simulador de Partido en Tiempo Real (Digital Twin)")
        st.markdown("Planificador táctico automatizado contra el rival de la jornada, simulador estocástico minuto a minuto (estilo NFL) y laboratorio Monte Carlo.")

        cp1, cp2 = st.columns(2)
        with cp1:
            local_t = st.selectbox("Tu Equipo (Local):", available_clubs, index=def_club_idx, key="mod3_local_team")
        with cp2:
            rival_options = [c for c in available_clubs if c != local_t] or available_clubs
            def_rival_idx = rival_options.index("Real Madrid") if "Real Madrid" in rival_options else 0
            rival_t = st.selectbox("Equipo Rival (Visitante):", rival_options, index=def_rival_idx, key="mod3_rival_team")

        local_squad = df_players[df_players["team_name"] == local_t]
        rival_squad = df_players[df_players["team_name"] == rival_t]

        tab_plan, tab_sim, tab_whatif = st.tabs([
            "📋 Pizarra Táctica vs Rival",
            "⏱️ Simulador de Partido (Digital Twin)",
            "🎲 Laboratorio 'What-If' Monte Carlo"
        ])

        with tab_plan:
            st.markdown(f"#### 🛡️ Preparación de Pizarra Táctica: {local_t} vs {rival_t}")
            st.caption("Optimiza la alineación titular cruzando métricas individuales con debilidades posicionales del adversario.")
            
            if st.button("🧠 Generar Plan Táctico & 11 Titular Óptimo", key="mod3_btn_plan"):
                try:
                    with st.spinner(f"Analizando patrones tácticos de {rival_t}..."):
                        plan = desk.prepare_match_plan(local_t, rival_t)
                    
                    form_rec = plan.get("formation_recommended", "4-3-3 Dinámico")
                    instructions = plan.get("tactical_instructions", "Equilibrio táctico y transiciones rápidas.")
                    
                    st.markdown(f"### 🛡️ Formación Prescrita: `{form_rec}`")
                    st.info(f"💡 **Directriz del Míster**: {instructions}")
                    
                    optimal_lineup = plan.get("optimal_lineup")
                    if optimal_lineup is not None and isinstance(optimal_lineup, pd.DataFrame) and not optimal_lineup.empty:
                        cols_xi = [c for c in ["player_name", "position", "age", "tactical_fit_score", "xg_p90", "xa_p90", "interceptions_p90"] if c in optimal_lineup.columns]
                        st.dataframe(
                            optimal_lineup[cols_xi].style.background_gradient(subset=["tactical_fit_score"], cmap="Blues")
                            .format({"tactical_fit_score": "{:.3f}", "xg_p90": "{:.2f}", "xa_p90": "{:.2f}", "interceptions_p90": "{:.2f}"}),
                            use_container_width=True
                        )
                    
                    plays = plan.get("tactical_plays", [])
                    if plays:
                        st.markdown("##### 📐 Jugadas Recomendadas de Pizarra")
                        for p in plays:
                            st.caption(f"• **{p.get('categoria', 'Táctica')}**: {p.get('jugada', '')} — *{p.get('explicacion', '')}*")
                except Exception as e:
                    st.error(f"Error generando plan táctico: {e}")

        with tab_sim:
            st.markdown("#### ⏱️ Simulación Estocástica Minuto a Minuto (Digital Twin estilo NFL)")
            st.caption("Simula un encuentro completo en tiempo real calculando probabilidades dinámicas de victoria, ocasiones de peligro y goles esperados.")
            
            if st.button("▶️ Ejecutar Simulación Digital Twin", key="mod3_btn_sim"):
                try:
                    with st.spinner(f"Simulando 90 minutos de partido entre {local_t} y {rival_t}..."):
                        sim_out = scenario_sim.simulate_match_timeline(
                            local_t, rival_t,
                            base_elo_home=elo.get_rating(local_t),
                            base_elo_away=elo.get_rating(rival_t)
                        )
                    
                    final_score = sim_out.get("final_score", (1, 0))
                    st.success(f"🏆 Marcador Final: **{local_t} {final_score[0]} - {final_score[1]} {rival_t}**")
                    
                    sm1, sm2, sm3 = st.columns(3)
                    sm1.metric(f"xG Acumulado {local_t}", f"{sim_out.get('total_xg_home', 1.2):.2f}")
                    sm2.metric(f"xG Acumulado {rival_t}", f"{sim_out.get('total_xg_away', 0.9):.2f}")
                    sm3.metric("Ocasiones Clave Registradas", f"{len(sim_out.get('events', []))}")

                    # Gráfico de evolución temporal de xG
                    timeline_df = sim_out.get("timeline_df")
                    if timeline_df is not None and not timeline_df.empty:
                        fig_time = draw_digital_twin_xg_timeline(timeline_df, local_t, rival_t)
                        st.plotly_chart(fig_time, use_container_width=True)

                    # Relato de eventos
                    events = sim_out.get("events", [])
                    if events:
                        st.markdown("##### 📝 Línea de Tiempo de Eventos y Remates Clave")
                        ev_df = pd.DataFrame(events)
                        show_ev_cols = [c for c in ["minute", "team", "event_type", "xg_value", "win_prob_home", "desc"] if c in ev_df.columns]
                        st.dataframe(ev_df[show_ev_cols], use_container_width=True)
                except Exception as e:
                    st.error(f"Error en simulación estocástica Digital Twin: {e}")

        with tab_whatif:
            st.markdown("#### 🎲 Laboratorio 'What-If': Evaluación de Impacto de Nuevos Fichajes")
            st.caption("Comprueba mediante simulaciones de Monte Carlo cómo cambiarían las probabilidades de victoria antes de realizar un desembolso económico.")
            try:
                st.markdown(f"Predicción probabilística con ratings Elo calibrados (**{local_t}**: {elo.get_rating(local_t):.0f} vs **{rival_t}**: {elo.get_rating(rival_t):.0f}):")
                loc_lineup = local_squad.head(11) if len(local_squad) >= 11 else df_players.head(11)
                riv_lineup = rival_squad.head(11) if len(rival_squad) >= 11 else df_players.tail(11)
                
                mc_pred = desk.matchup_analyzer.predict_match_outcome(
                    local_t, rival_t,
                    local_lineup=loc_lineup,
                    rival_lineup=riv_lineup
                )
                
                col_m1, col_m2, col_m3 = st.columns(3)
                p_win_home = mc_pred.get('win_home_prob', 0.45) * 100
                p_draw = mc_pred.get('draw_prob', 0.28) * 100
                p_win_away = mc_pred.get('win_away_prob', 0.27) * 100
                
                col_m1.metric(f"Victoria {local_t}", f"{p_win_home:.1f}%")
                col_m2.metric("Empate", f"{p_draw:.1f}%")
                col_m3.metric(f"Victoria {rival_t}", f"{p_win_away:.1f}%")

                st.markdown("##### 💡 Simular Intercambio de Jugador (What-If Transfer Swap)")
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    p_out_options = sorted(local_squad["player_name"].unique()) if not local_squad.empty else ["Titular Base"]
                    sel_p_out = st.selectbox("Jugador a sustituir en tu XI:", p_out_options, index=0, key="whatif_p_out")
                with col_w2:
                    p_in_pool = df_players[df_players["team_name"] != local_t]
                    p_in_options = sorted(p_in_pool["player_name"].unique()) if not p_in_pool.empty else ["Candidato Mercado"]
                    sel_p_in = st.selectbox("Nuevo Fichaje a incorporar:", p_in_options, index=0, key="whatif_p_in")

                cand_matches = df_players[df_players["player_name"] == sel_p_in]
                out_matches = local_squad[local_squad["player_name"] == sel_p_out]
                
                if not cand_matches.empty and not out_matches.empty:
                    cand_row = cand_matches.iloc[0]
                    out_row = out_matches.iloc[0]
                    delta_xg = float(cand_row.get("xg_p90", 0.0)) - float(out_row.get("xg_p90", 0.0))
                    delta_xa = float(cand_row.get("xa_p90", 0.0)) - float(out_row.get("xa_p90", 0.0))
                    delta_mv = float(cand_row.get("market_value_m", 0.0)) - float(out_row.get("market_value_m", 0.0))

                    st.markdown(f"""
                    <div class="glass-card">
                        <p>• <b>Impacto Ofensivo Neto:</b> {delta_xg:+.2f} xG/90 y {delta_xa:+.2f} xA/90</p>
                        <p>• <b>Diferencial de Inversión:</b> {delta_mv:+.1f} M€</p>
                        <p>• <b>Efecto en Probabilidad de Victoria:</b> <b>{'+' if delta_xg > 0 else ''}{round(delta_xg * 18.0, 1)}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Aviso en simulación What-If: {e}")

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
            try:
                sb_comps = hub.statsbomb.get_featured_competitions()
                st.dataframe(sb_comps, use_container_width=True)
                st.markdown("##### 🏟️ Muestra de Eventos: Final Champions League (Tottenham vs Liverpool)")
                events = hub.statsbomb.get_match_events(match_id=22912)
                cols_ev = [c for c in ["minute", "type", "team", "location_x", "location_y"] if c in events.columns]
                st.dataframe(events[cols_ev].head(12), use_container_width=True)
            except Exception as e:
                st.warning(f"Aviso consultando StatsBomb Open Data: {e}")

        with t2:
            st.markdown("#### 🎯 Understat Shot Engine & Eficiencia de Finalización (G - xG)")
            st.caption("Matriz de remates tiro a tiro sobre media cancha reglamentaria con cálculo de la métrica de sobre-rendimiento.")
            try:
                shots = hub.understat.get_shot_data(limit=250)
                st.plotly_chart(draw_understat_shot_map(shots), use_container_width=True)
                eff_df = hub.understat.compute_player_finishing_efficiency(shots)
                if eff_df is not None and not eff_df.empty:
                    st.dataframe(
                        eff_df.style.background_gradient(subset=["goals_minus_xg", "finishing_ratio"], cmap="Greens")
                        .format({"total_xg": "{:.2f}", "goals_minus_xg": "{:+.2f}", "finishing_ratio": "{:.2f}x"}),
                        use_container_width=True
                    )
            except Exception as e:
                st.warning(f"Aviso en Understat Shot Engine: {e}")

        with t3:
            st.markdown("#### 📊 SoccerData & LanusStats (FBref & Sofascore p90 Metrics)")
            st.markdown("Métricas avanzadas normalizadas por 90 minutos: xAG, SCA, acarreos progresivos y presiones defensivas.")
            try:
                sd_df = hub.soccerdata.get_advanced_p90_metrics()
                if sd_df is not None and not sd_df.empty:
                    grad_cols = [c for c in ["xg_p90", "xag_p90", "sca_p90"] if c in sd_df.columns]
                    st_view = sd_df.style
                    if grad_cols:
                        st_view = st_view.background_gradient(subset=grad_cols, cmap="Blues")
                    st.dataframe(st_view, use_container_width=True)
            except Exception as e:
                st.warning(f"Aviso en SoccerData Hub: {e}")

        with t4:
            st.markdown("#### 🏴󠁧󠁢󠁥󠁮󠁧󠁿 API Oficial de Fantasy Premier League (FPL)")
            st.markdown("API pública oficial en JSON sin autenticación ni rate limits (`fantasy.premierleague.com/api/bootstrap-static/`). Conexión segura con `verify=False` y timeout de 6s.")
            try:
                fpl_df = hub.fpl.get_fpl_players(limit=25)
                if fpl_df is not None and not fpl_df.empty:
                    num_cols = ["ict_index", "influence", "creativity", "threat"]
                    fmt_fpl = {c: "{:.1f}" for c in num_cols if c in fpl_df.columns}
                    grad_fpl = [c for c in ["ict_index", "influence", "threat"] if c in fpl_df.columns]
                    st_fpl = fpl_df.style
                    if grad_fpl:
                        st_fpl = st_fpl.background_gradient(subset=grad_fpl, cmap="Purples")
                    st.dataframe(st_fpl.format(fmt_fpl), use_container_width=True)
            except Exception as e:
                st.warning(f"Aviso consultando FPL API: {e}")

        with t5:
            st.markdown("#### 💶 Transfermarkt Datalake & Capology (Contratos y Salarios)")
            st.markdown("Oportunidades de mercado por vencimiento contractual inminente (< 12 meses) y amortizaciones de masa salarial.")
            try:
                tm_opps = hub.transfermarkt.get_moneyball_contract_opportunities()
                if tm_opps is not None and not tm_opps.empty:
                    fmt_tm = {}
                    if "market_value_m" in tm_opps.columns: fmt_tm["market_value_m"] = "{:.1f} M€"
                    if "estimated_transfer_fee_m" in tm_opps.columns: fmt_tm["estimated_transfer_fee_m"] = "{:.1f} M€"
                    if "expected_saving_m" in tm_opps.columns: fmt_tm["expected_saving_m"] = "{:.1f} M€"
                    if "opportunity_score" in tm_opps.columns: fmt_tm["opportunity_score"] = "{:.1f}"

                    grad_tm = [c for c in ["opportunity_score", "expected_saving_m"] if c in tm_opps.columns]
                    st_tm = tm_opps.style
                    if grad_tm:
                        st_tm = st_tm.background_gradient(subset=grad_tm, cmap="Greens")
                    st.dataframe(st_tm.format(fmt_tm), use_container_width=True)
            except Exception as e:
                st.warning(f"Aviso en Transfermarkt Hub: {e}")

        with t6:
            st.markdown("#### 🧬 Kloppy (PySport Universal Standardization)")
            st.markdown("Demostración de ingeniería de datos: normalización de flujos heterogéneos de eventos a coordenadas universales $[0, 100]$ y taxonomía PySport.")
            try:
                shots = hub.understat.get_shot_data(limit=10)
                sample_evs = shots.rename(columns={"x": "location_x", "y": "location_y", "result": "type"})
                std_evs = hub.kloppy.standardize_event_stream(sample_evs)
                st.dataframe(std_evs, use_container_width=True)
            except Exception as e:
                st.warning(f"Aviso en Kloppy Standardizer: {e}")

        with t7:
            st.markdown("#### 🔄 Persistencia y Sincronización en PostgreSQL 16")
            st.markdown(f"Sincroniza y persiste los datos multitemporada de los futbolistas y plantillas de la red mundial ({clean_season}) en las tablas relacionales de PostgreSQL.")
            if st.button("📥 Ejecutar Sincronización con PostgreSQL", key="mod4_btn_pg_sync"):
                try:
                    with st.spinner("Sincronizando registros con PostgreSQL..."):
                        synced = sync_global_database_to_postgres(df_players)
                    if synced > 0:
                        st.success(f"✅ Sincronización completada: {synced} futbolistas actualizados en PostgreSQL.")
                    else:
                        st.warning("⚠️ No se pudo completar la sincronización con PostgreSQL. Verifique si el contenedor `fb_postgres` está activo.")
                except Exception as e:
                    st.error(f"Error durante la sincronización relacional con PostgreSQL: {e}")


if __name__ == "__main__":
    main()
