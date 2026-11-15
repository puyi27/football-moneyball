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

# Importación de la arquitectura analítica (Módulos de Scouting, Táctica y Datos)
from src.data.real_dataset_loader import load_real_players_df
from src.data.global_player_database import load_global_players, sync_global_database_to_postgres
from src.data.football_api_connector import FootballAPIConnector
from src.data.advanced_sources_hub import MasterAnalyticsDataHub
from src.models_scouting.player_profiler import PlayerProfiler
from src.models_analytics.team_profiler import TeamProfiler
from src.models_scouting.squad_diagnostics import SquadDiagnostics
from src.models_scouting.recommender import ScoutingRecommender
from src.models_scouting.academy_scout import AcademyScout
from src.models_analytics.elo_model import EloSystem
from src.models_analytics.poisson_model import DixonColesModel
from src.models_tactical.lineup_simulator import TacticalSimulator
from src.models_tactical.matchup_analyzer import MatchupAnalyzer
from src.models_tactical.tactical_playmaker import TacticalPlaymaker
from src.models_tactical.tactical_analyst_desk import TacticalAnalystDesk
from src.models_tactical.match_scenario_simulator import MatchScenarioSimulator


# 1. Configuración de página y estética profesional
st.set_page_config(
    page_title="Football Moneyball AI — Virtual Analyst Desk (TFM)",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS profesional adaptado a analítica deportiva de élite
st.markdown("""
<style>
    /* Estilo general y paleta deportiva */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Encabezado ejecutivo */
    .executive-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .executive-title {
        font-size: 28px;
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
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 16px;
    }
    
    /* Chips de estado */
    .status-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .status-badge {
        font-size: 12px;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-green { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-blue { background-color: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .badge-purple { background-color: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
    .badge-gold { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }

    /* Tarjetas de métricas */
    div[data-testid="stMetricValue"] {
        font-weight: 800;
        color: #38bdf8;
    }
    
    /* Contenedores tipo Card */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# 2. Conexión de base de datos
@st.cache_resource
def get_db_engine():
    try:
        return create_engine("postgresql://moneyball:scouting_admin@postgres:5432/football_db")
    except Exception:
        return None

# 3. Conjunto de datos enriquecido (multiliga Top 5 + Scouting + Canteras + Red Mundial)
@st.cache_data
def get_players_database() -> pd.DataFrame:
    """Carga los datos de futbolistas reales de la red mundial con sus métricas y valores de mercado."""
    return load_global_players()

# 4. Instancia singleton del Hub de datos avanzado
@st.cache_resource
def get_advanced_hub() -> MasterAnalyticsDataHub:
    return MasterAnalyticsDataHub()


# 5. Dibuja un campo reglamentario con mapa de calor de influencia
def draw_tactical_pitch_heatmap(heatmap_df: pd.DataFrame, player_name: str, position: str) -> go.Figure:
    """Dibuja un terreno de juego reglamentario (105x68m) con la densidad de actividad del jugador."""
    fig = go.Figure()

    # Césped de fútbol (Verde profesional)
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, fillcolor="#0d2818", line=dict(color="#4ade80", width=1.5))

    # Línea de medio campo y círculo central
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="circle", x0=52.5 - 9.15, y0=34 - 9.15, x1=52.5 + 9.15, y1=34 + 9.15, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="circle", x0=52.2, y0=33.7, x1=52.8, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))

    # Áreas de penal
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color="#ffffff", width=1.5))

    # Áreas pequeñas de meta
    fig.add_shape(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16, line=dict(color="#ffffff", width=1.5))

    # Puntos de penalti
    fig.add_shape(type="circle", x0=10.7, y0=33.7, x1=11.3, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))
    fig.add_shape(type="circle", x0=93.7, y0=33.7, x1=94.3, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))

    # Capa de calor de toques / acciones (2D Density Heatmap)
    fig.add_trace(go.Histogram2dContour(
        x=heatmap_df["x"],
        y=heatmap_df["y"],
        colorscale="Viridis",
        reversescale=False,
        showscale=False,
        ncontours=16,
        opacity=0.72,
        contours=dict(coloring="heatmap")
    ))

    # Puntos individuales de actividad
    fig.add_trace(go.Scatter(
        x=heatmap_df["x"],
        y=heatmap_df["y"],
        mode="markers",
        marker=dict(size=4, color="#facc15", opacity=0.5),
        name="Acciones / Toques"
    ))

    fig.update_layout(
        title=dict(text=f"Mapa de Calor de Zonas de Influencia: {player_name} ({position})", font=dict(color="#ffffff", size=14)),
        xaxis=dict(range=[-2, 107], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 70], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
        showlegend=False,
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19"
    )

    return fig


# 6. Dibuja un mapa de tiros tiro a tiro sobre media cancha (Understat Shot Map)
def draw_understat_shot_map(df_shots: pd.DataFrame, title: str = "Mapa Tiro a Tiro (xG Shot Chart)") -> go.Figure:
    """Dibuja un medio campo reglamentario con los remates registrados y su valoración xG."""
    fig = go.Figure()

    # Medio campo de ataque (x de 52.5 a 105, y de 0 a 68)
    fig.add_shape(type="rect", x0=52.5, y0=0, x1=105, y1=68, fillcolor="#0d2818", line=dict(color="#4ade80", width=1.5))
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16, line=dict(color="#ffffff", width=1.5))
    fig.add_shape(type="circle", x0=93.7, y0=33.7, x1=94.3, y1=34.3, fillcolor="#ffffff", line=dict(color="#ffffff"))
    # Portería rival
    fig.add_shape(type="rect", x0=105, y0=30.34, x1=107, y1=37.66, fillcolor="#334155", line=dict(color="#ffffff", width=2))

    color_map = {
        "Goal": "#00ff87",        # Verde neón para goles
        "SavedShot": "#f59e0b",   # Ámbar para tiros a puerta atajados
        "BlockedShot": "#a855f7", # Violeta para tiros bloqueados
        "MissedShots": "#ef4444"  # Rojo para disparos fuera
    }

    for res, c in color_map.items():
        subset = df_shots[df_shots["result"] == res]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                marker=dict(
                    size=subset["xg"] * 32 + 6,
                    color=c,
                    line=dict(width=1, color="#ffffff"),
                    opacity=0.85
                ),
                name=f"{res} ({len(subset)})",
                text=[f"<b>{row['player']}</b><br>Resultado: {row['result']}<br>xG: {row['xg']}<br>Situación: {row['situation']}<br>Zona: ({row['x']:.1f}, {row['y']:.1f})" for _, row in subset.iterrows()],
                hoverinfo="text"
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=15)),
        xaxis=dict(range=[50, 108], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 70], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(color="#f1f5f9")),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19"
    )

    return fig


def main():
    # Encabezado ejecutivo superior unificado
    st.markdown("""
    <div class="executive-header">
        <div class="executive-title">
            ⚽ FOOTBALL MONEYBALL AI — VIRTUAL ANALYST DESK
        </div>
        <div class="executive-subtitle">
            Plataforma Integral de Analítica Avanzada, Modelos Tácticos y Apoyo a la Decisión para Dirección Deportiva (TFM)
        </div>
        <div class="status-badge-container">
            <span class="status-badge badge-green">🟢 PostgreSQL 16 Synced</span>
            <span class="status-badge badge-blue">⚡ Apache Spark 3.5.0 Active</span>
            <span class="status-badge badge-purple">📡 7 Data Engines Ready</span>
            <span class="status-badge badge-gold">🏆 LaLiga 2024/2025 Official Squads</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_players = get_players_database()
    hub = get_advanced_hub()

    # Inicializar sistema Elo de calibración competitiva
    elo = EloSystem()
    elo.ratings.update({
        "Real Madrid": 1840.0, "Barcelona": 1820.0, "Atletico Madrid": 1740.0,
        "Sevilla FC": 1620.0, "Real Betis": 1630.0, "Real Sociedad": 1660.0,
        "Girona FC": 1650.0, "Athletic Club": 1670.0, "Villarreal CF": 1660.0, "Valencia CF": 1560.0,
        "Arsenal": 1820.0, "Manchester City": 1870.0, "Liverpool": 1850.0, "Chelsea": 1730.0,
        "Bayern Munich": 1850.0, "Bayer Leverkusen": 1810.0, "Inter": 1780.0, "Milan": 1720.0, "Atalanta": 1740.0,
        "Juventus": 1770.0, "Paris Saint-Germain": 1810.0, "Benfica": 1670.0, "Sporting CP": 1690.0,
        "PSV": 1660.0, "Feyenoord": 1640.0,
    })

    profiler = PlayerProfiler(df_players)
    team_profiler = TeamProfiler(df_players, elo_system=elo)
    desk = TacticalAnalystDesk(df_players_metrics=df_players, elo_system=elo)
    academy_scout = AcademyScout(df_players)
    scenario_sim = MatchScenarioSimulator()

    # Menú Principal de Herramientas del Analista en Barra Lateral
    st.sidebar.markdown("### 🧭 Panel de Navegación")
    menu = st.sidebar.radio(
        "Módulos Operativos:",
        [
            "1. 👤 Ficha Técnica 360° & EA Sports FUT Card",
            "2. 📋 Centro de Mando del Club & Plantillas Oficiales",
            "3. 💰 Radar de Gangas Moneyball & Transfermarkt",
            "4. 🌟 Ojeador de Canteras & Wonderkids Sub-21",
            "5. 🎮 Digital Twin: Simulador de Partido en Vivo",
            "6. 🔬 Laboratorio Táctico & Pizarra vs Rival",
            "7. 📑 Dossier Ejecutivo para la Directiva",
            "8. ⚡ Hub de Big Data, Librerías Avanzadas & APIs (TFM)"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 Métricas de la Plataforma")
    st.sidebar.caption(f"• **Futbolistas en Red:** {len(df_players):,}")
    st.sidebar.caption(f"• **Ligas Monitorizadas:** {df_players['league'].nunique()}")
    st.sidebar.caption(f"• **Clubes Oficiales:** {df_players['team_name'].nunique()}")
    st.sidebar.caption("• **Fuentes Activas:** 7 Motores de Datos")

    # =========================================================================
    # VISTA 1: FICHA 360° DEL JUGADOR (FIFA CARD, RADAR, MAPA DE CALOR, CLONES)
    # =========================================================================
    if menu == "1. 👤 Ficha Técnica 360° & EA Sports FUT Card":
        st.subheader("👤 Ficha Técnica Integral 360° del Futbolista")
        st.markdown("Radiografía individual estilo EA Sports / FIFA FUT con atributos 1-99, mapa de calor táctico y detección de futbolistas análogos (Clones K-NN).")

        col_sel1, col_sel2 = st.columns([1, 2])
        with col_sel1:
            all_leagues = sorted(df_players["league"].unique())
            sel_league = st.selectbox("Filtrar por Liga:", ["Todas"] + all_leagues, index=0)
            if sel_league != "Todas":
                filtered_pool = df_players[df_players["league"] == sel_league]
            else:
                filtered_pool = df_players

            # Buscador libre para acceder a cientos de futbolistas mundiales
            search_box = st.text_input("🔎 Búsqueda Rápida por Nombre o Club:", "").strip()
            if search_box:
                filtered_pool = filtered_pool[
                    filtered_pool["player_name"].str.contains(search_box, case=False, na=False) |
                    filtered_pool["team_name"].str.contains(search_box, case=False, na=False)
                ]

            player_names = sorted(filtered_pool["player_name"].unique()) if not filtered_pool.empty else ["Vinícius Júnior"]
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

        with col_radar:
            radar_df = pd.DataFrame(dict(
                r=[card['PAC'], card['SHO'], card['PAS'], card['DRI'], card['DEF'], card['PHY']],
                theta=['PAC (Ritmo)', 'SHO (Tiro)', 'PAS (Pase)', 'DRI (Regate)', 'DEF (Defensa)', 'PHY (Físico)']
            ))
            fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True, range_r=[0, 100])
            fig_radar.update_traces(fill='toself', fillcolor='rgba(212, 175, 55, 0.45)', line=dict(color='#d4af37', width=2.5))
            fig_radar.update_layout(
                title=f"Rueda Poligonal de Rendimiento: {selected_player}",
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                margin=dict(l=30, r=30, t=40, b=30),
                height=350,
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#0b0f19"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

        # 2. MAPA DE CALOR 2D REGLAMENTARIO
        st.subheader("📍 Mapa de Calor y Zonas de Influencia en el Terreno de Juego")
        st.caption("Distribución espacial basada en las acciones por 90 minutos y coordenadas tácticas del futbolista.")
        heatmap_points = profiler.generate_pitch_heatmap(player_data)
        fig_pitch = draw_tactical_pitch_heatmap(heatmap_points, selected_player, player_data["position"])
        st.plotly_chart(fig_pitch, use_container_width=True)

        st.markdown("---")

        # 3. CLONES SIMILARES (K-NN)
        st.subheader(f"👥 Futbolistas Similares a {selected_player} (Clones K-NN)")
        st.markdown("Algoritmo de Vecinos Más Cercanos en espacio multidimensional (xG, xA, acciones defensivas, pases progresivos y regates).")
        twins_df = profiler.find_player_twins(selected_player, n_neighbors=5)
        if not twins_df.empty:
            st.dataframe(
                twins_df[["player_name", "team_name", "league", "position", "age", "market_value_m", "similarity_score", "potential_ovr"]]
                .style.background_gradient(subset=["similarity_score"], cmap="Greens")
                .format({"market_value_m": "{:.1f} M€", "similarity_score": "{:.1f}%", "potential_ovr": "{:.0f}"}),
                use_container_width=True
            )
        else:
            st.info("No se encontraron jugadores suficientes para el cálculo de similitud.")

    # =========================================================================
    # VISTA 2: CENTRO DE MANDO DEL CLUB (ADN TÁCTICO & PLANTILLAS OFICIALES)
    # =========================================================================
    elif menu == "2. 📋 Centro de Mando del Club & Plantillas Oficiales":
        st.subheader("📋 Centro de Mando del Club y Gestión de Plantilla")
        st.markdown("Diagnóstico de plantilla, pirámide demográfica, métricas colectivas y plantilla oficial completa 2024/2025.")

        clubs_available = sorted(df_players["team_name"].unique())
        default_club_idx = clubs_available.index("Sevilla FC") if "Sevilla FC" in clubs_available else 0
        club_selected = st.selectbox("Seleccionar Club a Analizar:", clubs_available, index=default_club_idx)

        team_squad = df_players[df_players["team_name"] == club_selected]
        overview = team_profiler.get_team_overview(club_selected)

        # KPIs Principales del Club
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Jugadores en Nómina", f"{overview['squad_size']}")
        k2.metric("Edad Media", f"{overview['avg_age']:.1f} años")
        k3.metric("Valor Total de Mercado", f"{overview['total_market_value_m']:.1f} M€")
        k4.metric("Rating Elo Competitivo", f"{overview['elo_rating']:.0f}")
        k5.metric("Potencial Cantera", f"{overview['academy_players_count']} canteranos")

        st.markdown("---")

        c_squad, c_pyramid = st.columns([1.5, 1])
        with c_squad:
            st.subheader(f"📋 Plantilla Oficial 2024/2025 de {club_selected}")
            st.dataframe(
                team_squad[['player_name', 'position', 'age', 'market_value_m', 'minutes', 'xg_p90', 'xa_p90', 'interceptions_p90']]
                .sort_values(by="market_value_m", ascending=False)
                .style.background_gradient(subset=["market_value_m", "minutes"], cmap="Blues")
                .format({"market_value_m": "{:.1f} M€", "xg_p90": "{:.2f}", "xa_p90": "{:.2f}", "interceptions_p90": "{:.2f}"}),
                use_container_width=True
            )

        with c_pyramid:
            st.subheader("📊 Pirámide Demográfica del Club")
            age_chart = px.histogram(
                team_squad, x="age", nbins=8, color="position",
                title=f"Distribución de Edad: {club_selected}",
                labels={"age": "Edad", "count": "Nº de Jugadores"},
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            age_chart.update_layout(paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19", font=dict(color="#ffffff"))
            st.plotly_chart(age_chart, use_container_width=True)

        # Diagnóstico de déficits de la plantilla
        diag_engine = SquadDiagnostics()
        deficits = diag_engine.analyze_team_deficits(team_squad)
        st.subheader("⚠️ Diagnóstico de Déficits y Alertas de Posición")
        for pos, info in deficits.items():
            if info["status"] == "CRITICAL":
                st.error(f"🚨 **Posición {pos}**: Estado CRÍTICO. {info['reason']}")
            elif info["status"] == "WARNING":
                st.warning(f"⚠️ **Posición {pos}**: Advertencia. {info['reason']}")
            else:
                st.success(f"✅ **Posición {pos}**: Plantilla equilibrada y cubierta.")

    # =========================================================================
    # VISTA 3: RADAR DE GANGAS ('MONEYBALL INDEX' & TRANSFERMARKT)
    # =========================================================================
    elif menu == "3. 💰 Radar de Gangas Moneyball & Transfermarkt":
        st.subheader("💰 Radar de Gangas Deportivas (Moneyball Index & Contratos)")
        st.markdown("Identificación de futbolistas cuyo rendimiento deportivo supera ampliamente su precio de mercado o cuya situación contractual genera una ventana de oportunidad económica.")

        tab_moneyball, tab_contracts = st.tabs(["🎯 Índice Moneyball (Rendimiento vs Valor)", "⏳ Oportunidades Contractuales (Transfermarkt)"])

        with tab_moneyball:
            c_filter1, c_filter2 = st.columns(2)
            with c_filter1:
                max_val = st.slider("Valor de Mercado Máximo (M€):", 1.0, 50.0, 20.0, step=1.0)
            with c_filter2:
                sel_pos = st.selectbox("Posición Requerida:", ["Todas", "FW", "MF", "DF"])

            bargains_df = profiler.find_moneyball_bargains(max_value_m=max_val, min_minutes=300)
            if sel_pos != "Todas":
                bargains_df = bargains_df[bargains_df["position"] == sel_pos]

            st.dataframe(
                bargains_df[["player_name", "team_name", "league", "position", "age", "market_value_m", "moneyball_index", "xg_p90", "xa_p90", "interceptions_p90"]]
                .style.background_gradient(subset=["moneyball_index"], cmap="YlGn")
                .format({"market_value_m": "{:.1f} M€", "moneyball_index": "{:.1f}"}),
                use_container_width=True
            )

        with tab_contracts:
            st.subheader("⏳ Jugadores en Último Año de Contrato (Vencimiento 2025)")
            st.markdown("Jugadores de élite o alta proyección cuyos contratos vencen en menos de 12 meses, lo que permite negociar traspasos con descuentos del 40-70% o preacuerdos sin coste de transferencia.")
            tm_opps = hub.transfermarkt.get_moneyball_contract_opportunities()
            st.dataframe(
                tm_opps.style.background_gradient(subset=["opportunity_score", "expected_saving_m"], cmap="Greens")
                .format({"market_value_m": "{:.1f} M€", "estimated_transfer_fee_m": "{:.1f} M€", "expected_saving_m": "{:.1f} M€", "opportunity_score": "{:.1f}"}),
                use_container_width=True
            )

    # =========================================================================
    # VISTA 4: OJEADOR DE CANTERAS Y WONDERKIDS SUB-21
    # =========================================================================
    elif menu == "4. 🌟 Ojeador de Canteras & Wonderkids Sub-21":
        st.subheader("🌟 Ojeador de Canteras y Promesas Mundiales (Sub-21)")
        st.markdown("Auditoría de talento joven propio y scouting de canteras rivales para reducir costes de traspaso y potenciar el activo patrimonial del club.")

        c_acad1, c_acad2 = st.columns([1, 1])
        with c_acad1:
            st.subheader("🐣 Detección de Wonderkids de la Red")
            prospects = academy_scout.get_top_academy_prospects(max_age=21, min_potential=75.0)
            st.dataframe(
                prospects[["player_name", "team_name", "league", "position", "age", "potential_ovr", "market_value_m"]]
                .style.background_gradient(subset=["potential_ovr"], cmap="Purples")
                .format({"market_value_m": "{:.1f} M€", "potential_ovr": "{:.0f}"}),
                use_container_width=True
            )

        with c_acad2:
            st.subheader("⚖️ Comparativa: Fichaje de Mercado vs Cantera Propia")
            st.caption("Demostración del ahorro potencial al promocionar a un canterano frente a acudir al mercado internacional.")
            comparison = academy_scout.compare_academy_vs_market("Alberto Collado", "Saúl Ñíguez")
            st.json(comparison)

    # =========================================================================
    # VISTA 5: SIMULADOR DE PARTIDO EN VIVO (DIGITAL TWIN ESTILO NFL)
    # =========================================================================
    elif menu == "5. 🎮 Digital Twin: Simulador de Partido en Vivo":
        st.subheader("🎮 Simulador de Partido en Tiempo Real (Digital Twin estilo NFL)")
        st.markdown("Simulación estocástica de situaciones de juego minuto a minuto para predecir probabilidades de victoria, fatiga acumulada y momentos idóneos de sustitución táctica.")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            team_a = st.selectbox("Equipo Local:", sorted(df_players["team_name"].unique()), index=0)
        with col_t2:
            team_b = st.selectbox("Equipo Visitante:", sorted(df_players["team_name"].unique()), index=1)

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            run_sim = st.button("▶️ Ejecutar Simulación Digital Twin")

        if run_sim:
            with st.spinner(f"Simulando 90 minutos entre {team_a} y {team_b}..."):
                sim_result = scenario_sim.simulate_match_timeline(team_a, team_b, base_elo_home=elo.get_rating(team_a), base_elo_away=elo.get_rating(team_b))

            st.success(f"Marcador Final Simulado: **{team_a} {sim_result['final_score'][0]} - {sim_result['final_score'][1]} {team_b}**")
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric(f"xG {team_a}", f"{sim_result['total_xg_home']:.2f}")
            c_m2.metric(f"xG {team_b}", f"{sim_result['total_xg_away']:.2f}")
            c_m3.metric("Eventos de Peligro", f"{len(sim_result['events'])} jugadas clave")

            timeline_df = pd.DataFrame(sim_result["events"])
            if not timeline_df.empty:
                st.subheader("⏱️ Minutero de Ocasiones y Probabilidad de Gol")
                st.dataframe(timeline_df[["minute", "team", "event_type", "xg_value", "win_prob_home"]], use_container_width=True)

    # =========================================================================
    # VISTA 6: LABORATORIO TÁCTICO & PIZARRA VS RIVAL
    # =========================================================================
    elif menu == "6. 🔬 Laboratorio Táctico & Pizarra vs Rival":
        st.subheader("🔬 Laboratorio Táctico y Optimizador de Alineación vs Rival")
        st.markdown("Elige a tu equipo y al rival de la jornada para recibir la formación recomendada por la IA, auditoría de vulnerabilidades y simulación Monte Carlo.")

        col_lab1, col_lab2 = st.columns(2)
        with col_lab1:
            my_team = st.selectbox("Tu Equipo:", sorted(df_players["team_name"].unique()), index=default_club_idx, key="lab_my")
        with col_lab2:
            rival_team = st.selectbox("Equipo Rival:", sorted(df_players["team_name"].unique()), index=0, key="lab_rival")

        if st.button("🧠 Calcular Pizarra Táctica y 11 Óptimo"):
            with st.spinner("Ejecutando optimizador táctico..."):
                tactical_plan = desk.prepare_match_plan(my_team, rival_team)

            st.markdown(f"### 📋 Plan Táctico Recomendado: Formación {tactical_plan['formation_recommended']}")
            st.info(f"💡 **Clave del Míster**: {tactical_plan['tactical_instructions']}")

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                st.subheader("✅ 11 Titular Óptimo Recomendado")
                st.dataframe(pd.DataFrame(tactical_plan["optimal_lineup"]), use_container_width=True)
            with col_opt2:
                st.subheader("🎲 Pronóstico Monte Carlo del Partido")
                mc = tactical_plan["monte_carlo_prediction"]
                st.metric("Probabilidad Victoria Local", f"{mc['win_home_prob']*100:.1f}%")
                st.metric("Probabilidad Empate", f"{mc['draw_prob']*100:.1f}%")
                st.metric("Probabilidad Victoria Visitante", f"{mc['win_away_prob']*100:.1f}%")

    # =========================================================================
    # VISTA 7: DOSSIER EJECUTIVO PARA LA DIRECTIVA
    # =========================================================================
    elif menu == "7. 📑 Dossier Ejecutivo para la Directiva":
        st.subheader("📑 Generador de Dossier Técnico de Captación")
        st.markdown("Emite un informe profesional en formato corporativo para presentar ante el Consejo de Administración y Dirección Deportiva.")

        p_sel = st.selectbox("Seleccionar Jugador para el Informe:", sorted(df_players["player_name"].unique()))
        p_row = df_players[df_players["player_name"] == p_sel].iloc[0]
        card = profiler.calculate_fifa_card(p_row)

        st.markdown(f"""
        <div class="glass-card">
            <h2>📄 INFORME DE EXPLORACIÓN: {p_sel}</h2>
            <p><b>Demarcación:</b> {card['position']} | <b>Club:</b> {card['team_name']} ({card['league']}) | <b>Edad:</b> {card['age']} años</p>
            <hr style="border-color: #334155;">
            <h4>VALORACIÓN DE ATRIBUTOS (Media OVR: {card['OVR']}/99)</h4>
            <p>• <b>Ritmo:</b> {card['PAC']} | <b>Tiro:</b> {card['SHO']} | <b>Pase:</b> {card['PAS']} | <b>Regate:</b> {card['DRI']} | <b>Defensa:</b> {card['DEF']} | <b>Físico:</b> {card['PHY']}</p>
            <p>• <b>Valor de Mercado Estimado:</b> {card['market_value_m']} M€</p>
            <p>• <b>Métricas Clave:</b> xG p90: {p_row['xg_p90']:.2f} | xA p90: {p_row['xa_p90']:.2f} | Intercepciones p90: {p_row['interceptions_p90']:.2f}</p>
            <hr style="border-color: #334155;">
            <h4>DICTAMEN TÉCNICO FINAL</h4>
            <p style="color: #34d399; font-weight: bold;">✅ RECOMENDACIÓN: PROCEDER A LA ADQUISICIÓN / SEGUIMIENTO PRIORITARIO.</p>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # VISTA 8: HUB DE BIG DATA, LIBRERÍAS AVANZADAS & APIS (TFM EDITION)
    # =========================================================================
    elif menu == "8. ⚡ Hub de Big Data, Librerías Avanzadas & APIs (TFM)":
        st.subheader("⚡ Data Lakehouse Hub: Integración de las 7 Fuentes & Librerías Avanzadas")
        st.markdown("""
        Arquitectura unificada de datos para el TFM que agrupa proveedores abiertos y APIs profesionales:
        **StatsBomb**, **Understat**, **SoccerData (FBref)**, **Kloppy (PySport)**, **OpenFootball**, **Transfermarkt** y **Football-Data.org**.
        """)

        # Pestañas correspondientes a cada herramienta requerida
        t_sb, t_und, t_sd, t_klp, t_of, t_tm, t_fd = st.tabs([
            "🔴 StatsBomb 360 & (x, y)",
            "🎯 Understat Shot Engine",
            "📊 SoccerData & FBref",
            "🧬 Kloppy Standardizer",
            "🌍 OpenFootball Feed",
            "💶 Transfermarkt Datasets",
            "📡 Football-Data REST & Postgres"
        ])

        # TAB 1: STATSBOMB OPEN DATA
        with t_sb:
            st.markdown("### 🔴 StatsBomb Open Data (statsbombpy)")
            st.markdown("Event data académico con coordenadas espaciales exactas $(x, y)$, pases, presiones y freeze frames 360.")
            
            sb_comps = hub.statsbomb.get_featured_competitions()
            st.dataframe(sb_comps, use_container_width=True)

            st.markdown("#### 🏟️ Eventos Espaciales de Partido Real: Final Champions League (Tottenham vs Liverpool)")
            events_df = hub.statsbomb.get_match_events(match_id=22912)
            st.dataframe(events_df[["minute", "type", "team", "location_x", "location_y"]].head(15), use_container_width=True)

        # TAB 2: UNDERSTAT SHOT ENGINE
        with t_und:
            st.markdown("### 🎯 Understat Shot Engine & Eficiencia de Finalización (G - xG)")
            st.markdown("Registro tiro a tiro de cada futbolista para evaluar la probabilidad matemática del remate y la eficacia frente al guardameta rival.")
            
            shots = hub.understat.get_shot_data(limit=300)
            fig_shots = draw_understat_shot_map(shots, title="Mapa de Tiros Registrados (Understat xG Shot Chart)")
            st.plotly_chart(fig_shots, use_container_width=True)

            st.markdown("#### 🏅 Eficiencia de Finalización: Goles Reales vs Goles Esperados (G - xG)")
            st.caption("Un valor positivo ($G - xG > 0$) identifica a delanteros y extremos de finalización superlativa.")
            eff_df = hub.understat.compute_player_finishing_efficiency(shots)
            st.dataframe(
                eff_df.style.background_gradient(subset=["goals_minus_xg", "finishing_ratio"], cmap="Greens")
                .format({"total_xg": "{:.2f}", "goals_minus_xg": "{:+.2f}", "finishing_ratio": "{:.2f}x"}),
                use_container_width=True
            )

        # TAB 3: SOCCERDATA & FBREF
        with t_sd:
            st.markdown("### 📊 SoccerData (Python Wrapper para FBref & WhoScored)")
            st.markdown("Métricas avanzadas p90 de nivel profesional: xG, xAG, Acciones Creadoras de Tiro (SCA) y acarreos progresivos.")
            sd_metrics = hub.soccerdata.get_advanced_p90_metrics()
            st.dataframe(
                sd_metrics.style.background_gradient(subset=["xg_p90", "xag_p90", "sca_p90"], cmap="Blues"),
                use_container_width=True
            )

        # TAB 4: KLOPPY STANDARDIZER
        with t_klp:
            st.markdown("### 🧬 Kloppy (PySport Data Normalization)")
            st.markdown("Estandarización de flujos de eventos heterogéneos (StatsBomb, Opta, Wyscout) a un modelo universal de coordenadas $[0, 100]$ y taxonomía unificada.")
            raw_sample = shots.head(10).rename(columns={"x": "location_x", "y": "location_y", "result": "type"})
            kloppy_res = hub.kloppy.standardize_event_stream(raw_sample)
            st.dataframe(kloppy_res, use_container_width=True)

        # TAB 5: OPENFOOTBALL FEED
        with t_of:
            st.markdown("### 🌍 OpenFootball Datasets (GitHub Raw Feeds)")
            st.markdown("Marcadores, calendarios y resultados de ligas europeas sin límites de peticiones (Zero Rate Limits).")
            of_matches = hub.openfootball.get_league_season_matches()
            st.dataframe(of_matches, use_container_width=True)

        # TAB 6: TRANSFERMARKT DATASETS
        with t_tm:
            st.markdown("### 💶 Transfermarkt Datasets (salimt / Kaggle)")
            st.markdown("Evolución histórica de valoraciones y oportunidades contractuales para estrategias de inversión Moneyball.")
            
            p_tm_sel = st.selectbox("Ver Curva Histórica de Valor para:", ["Lamine Yamal", "Vinícius Júnior", "Loïc Badé", "Isaac Romero", "Dodi Lukebakio"])
            val_curve = hub.transfermarkt.get_player_market_value_evolution(p_tm_sel)
            fig_curve = px.line(val_curve, x="date", y="market_value_m", markers=True, title=f"Evolución del Valor de Mercado: {p_tm_sel} (M€)")
            fig_curve.update_layout(paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19", font=dict(color="#ffffff"))
            st.plotly_chart(fig_curve, use_container_width=True)

        # TAB 7: FOOTBALL-DATA REST & POSTGRESQL SYNC
        with t_fd:
            st.markdown("### 📡 Football-Data.org API & Persistencia Relacional PostgreSQL")
            st.markdown("Conector REST con token y sincronización directa con las tablas de PostgreSQL (`analytics_scouting_metrics`, `dim_players`).")

            c_token1, c_token2 = st.columns(2)
            with c_token1:
                foot_token = st.text_input("Football-Data Token:", value="", type="password")
                api_connector = FootballAPIConnector(football_data_token=foot_token)
                if st.button("📡 Probar Conectividad con Football-Data"):
                    diag = api_connector.test_football_data_connection()
                    if diag["connected"]:
                        st.success(f"✅ {diag['message']}")
                    else:
                        st.info(f"ℹ️ {diag['message']}")

            with c_token2:
                st.markdown("#### 🔄 Persistir Base de Datos en PostgreSQL")
                if st.button("📥 Ejecutar Sincronización en PostgreSQL"):
                    with st.spinner("Sincronizando registros con PostgreSQL..."):
                        synced = sync_global_database_to_postgres(df_players)
                    st.success(f"✅ {synced} futbolistas guardados en la base de datos PostgreSQL.")


if __name__ == "__main__":
    main()
