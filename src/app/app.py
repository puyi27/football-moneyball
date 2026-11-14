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
def load_all_players_data() -> pd.DataFrame:
    engine = get_db_engine()
    if engine:
        try:
            return pd.read_sql("SELECT * FROM analytics_scouting_metrics", engine)
        except Exception:
            pass

    # Generador sintético realista de respaldo con cobertura Top 5 y Scouting
    np.random.seed(42)
    teams_leagues = {
        # LaLiga
        "Real Madrid": "LaLiga", "Barcelona": "LaLiga", "Atletico Madrid": "LaLiga",
        "Sevilla FC": "LaLiga", "Real Betis": "LaLiga", "Real Sociedad": "LaLiga",
        "Girona FC": "LaLiga", "Athletic Club": "LaLiga", "Valencia CF": "LaLiga",
        # Premier League
        "Arsenal": "Premier League", "Manchester City": "Premier League", "Liverpool": "Premier League",
        "Aston Villa": "Premier League", "Brighton": "Premier League",
        # Serie A
        "Inter": "Serie A", "Juventus": "Serie A", "Milan": "Serie A", "Atalanta": "Serie A",
        # Bundesliga
        "Bayern Munich": "Bundesliga", "Bayer Leverkusen": "Bundesliga", "Borussia Dortmund": "Bundesliga",
        # Ligue 1
        "Paris Saint-Germain": "Ligue 1", "Monaco": "Ligue 1", "Lille": "Ligue 1",
        # Ligas de Desarrollo / Talento
        "Benfica": "Liga Portugal", "Sporting CP": "Liga Portugal", "Porto": "Liga Portugal",
        "Ajax": "Eredivisie", "PSV": "Eredivisie", "Feyenoord": "Eredivisie",
    }

    positions = ["GK", "DF", "MF", "FW"]
    rows = []
    pid = 1

    for team, league in teams_leagues.items():
        n_players = 18
        for i in range(n_players):
            pos = positions[i % len(positions)]
            age = int(np.random.randint(18, 35))
            mins = int(np.random.randint(400, 2900))
            is_rel = mins >= 450

            if pos == "FW":
                xg = round(float(np.random.uniform(0.15, 0.85)), 2)
                xa = round(float(np.random.uniform(0.08, 0.40)), 2)
                interc = round(float(np.random.uniform(0.2, 0.8)), 2)
                tkl = round(float(np.random.uniform(0.2, 0.9)), 2)
            elif pos == "MF":
                xg = round(float(np.random.uniform(0.05, 0.35)), 2)
                xa = round(float(np.random.uniform(0.10, 0.50)), 2)
                interc = round(float(np.random.uniform(0.8, 2.2)), 2)
                tkl = round(float(np.random.uniform(0.8, 2.5)), 2)
            elif pos == "DF":
                xg = round(float(np.random.uniform(0.01, 0.10)), 2)
                xa = round(float(np.random.uniform(0.02, 0.18)), 2)
                interc = round(float(np.random.uniform(1.2, 2.8)), 2)
                tkl = round(float(np.random.uniform(1.2, 3.0)), 2)
            else:  # GK
                xg, xa = 0.0, 0.0
                interc, tkl = 0.1, 0.1

            rows.append({
                "player_id": pid,
                "player_name": f"{team} {pos}-{i+1}",
                "team_name": team,
                "league": league,
                "position": pos,
                "age": age,
                "minutes": mins,
                "xg_p90": xg,
                "xa_p90": xa,
                "interceptions_p90": interc,
                "tackles_p90": tkl,
                "goals_p90": xg * 0.9,
                "assists_p90": xa * 0.8,
                "is_reliable": is_rel,
            })
            pid += 1

    return pd.DataFrame(rows)


def main():
    st.sidebar.title("⚽ Football Moneyball AI")
    st.sidebar.caption("Virtual Analyst Desk & What-If Engine (TFM)")
    st.sidebar.markdown("---")

    df_all = load_all_players_data()

    # Inicializar sistema Elo de calibración
    elo = EloSystem()
    elo.ratings.update({
        "Real Madrid": 1840.0, "Barcelona": 1820.0, "Atletico Madrid": 1740.0,
        "Sevilla FC": 1620.0, "Real Betis": 1630.0, "Real Sociedad": 1660.0,
        "Girona FC": 1650.0, "Athletic Club": 1670.0, "Valencia CF": 1560.0,
        "Arsenal": 1820.0, "Manchester City": 1870.0, "Bayern Munich": 1850.0,
        "Inter": 1780.0, "Paris Saint-Germain": 1810.0, "Benfica": 1670.0, "Sporting CP": 1680.0,
    })

    desk = TacticalAnalystDesk(df_players_metrics=df_all, elo_system=elo)

    # Navegación del Desk en 4 módulos operativos
    menu = st.sidebar.radio(
        "Navegación del Analista:",
        [
            "1. Auditoría y Scouting de Plantilla",
            "2. Laboratorio 'What-If' y Monte Carlo",
            "3. Preparación de Partido (Matchup vs Rival)",
            "4. Comparador de Perfiles (Radar p90)"
        ]
    )

    # =========================================================================
    # PESTAÑA 1: Auditoría de Plantilla y Scouting Multiliga
    # =========================================================================
    if menu == "1. Auditoría y Scouting de Plantilla":
        st.title("📋 Auditoría de Plantilla y Scouting de Fichajes")
        st.markdown("Identifica las posiciones deficitarias de tu equipo de LaLiga y localiza sustitutos en las 5 Grandes Ligas y competiciones de talento.")

        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            laliga_teams = sorted(df_all[df_all["league"] == "LaLiga"]["team_name"].unique())
            target_team = st.selectbox("Selecciona tu Equipo de LaLiga:", laliga_teams, index=laliga_teams.index("Sevilla FC") if "Sevilla FC" in laliga_teams else 0)
            eval_metric = st.selectbox("Métrica de Diagnóstico:", ["xg_p90", "xa_p90", "interceptions_p90"], index=0)

        with col_t2:
            audit = desk.audit_club_squad(team_name=target_team, metric=eval_metric)
            st.subheader(f"Diagnóstico de Posición para {target_team}")
            deficits_df = audit["deficits_table"]
            st.dataframe(deficits_df.style.applymap(lambda v: 'background-color: #5c1414' if v is True else '', subset=['needs_reinforcement']), use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Radar de Scouting Multiliga para Reforzar la Posición Débil")
        
        pos_to_scout = audit["weakest_position"] or "FW"
        c1, c2, c3 = st.columns(3)
        with c1:
            scout_pos = st.selectbox("Posición Objetivo a Reforzar:", ["FW", "MF", "DF"], index=["FW", "MF", "DF"].index(pos_to_scout) if pos_to_scout in ["FW", "MF", "DF"] else 0)
        with c2:
            leagues_available = list(df_all["league"].unique())
            selected_leagues = st.multiselect("Ligas a Ojear (Top 5 + Feeder):", leagues_available, default=["Premier League", "Serie A", "Bundesliga", "Ligue 1", "Liga Portugal", "Eredivisie"])
        with c3:
            max_scout_age = st.slider("Edad Máxima:", 18, 32, 26)

        candidates = desk.scout_transfer_targets(
            target_position=scout_pos,
            leagues_filter=selected_leagues,
            max_age=max_scout_age,
            top_k=8
        )

        if not candidates.empty:
            st.dataframe(candidates, use_container_width=True)
        else:
            st.warning("No se encontraron candidatos fiables con los filtros seleccionados.")

    # =========================================================================
    # PESTAÑA 2: Laboratorio de Simulación "What-If"
    # =========================================================================
    elif menu == "2. Laboratorio 'What-If' y Monte Carlo":
        st.title("🧪 Laboratorio de Simulación Táctica y Contexto ('What-If')")
        st.markdown("Simula el impacto de incorporar a un jugador recomendado en lugar de tu titular: recalcula $\\lambda$, $\\mu$ y proyecta los puntos finales vía Monte Carlo.")

        col1, col2 = st.columns(2)
        with col1:
            my_team = st.selectbox("Equipo:", sorted(df_all[df_all["league"] == "LaLiga"]["team_name"].unique()), index=0)
            team_players = df_all[(df_all["team_name"] == my_team) & (df_all["is_reliable"] == True)]
            player_out_name = st.selectbox("Jugador Titular a Sustituir:", team_players["player_name"].tolist())
            player_out_row = team_players[team_players["player_name"] == player_out_name].iloc[0]

        with col2:
            other_players = df_all[(df_all["team_name"] != my_team) & (df_all["is_reliable"] == True)]
            cand_name = st.selectbox("Fichaje / Candidato a Incorporar:", other_players["player_name"].tolist())
            cand_row = other_players[other_players["player_name"] == cand_name].iloc[0]

        st.markdown("---")
        if st.button("🚀 Ejecutar Simulación What-If y Monte Carlo (10.000 iteraciones)"):
            with st.spinner("Calculando matrices Dixon-Coles y simulación Monte Carlo..."):
                sim_res = desk.simulate_transfer_scenario(
                    team_name=my_team,
                    player_out_id=int(player_out_row["player_id"]),
                    candidate_player_id=int(cand_row["player_id"]),
                    base_lambda=1.25,
                    base_mu=1.15,
                    remaining_matches=15,
                    current_points=38,
                )

            swap = sim_res["swap_impact"]
            mc = sim_res["monte_carlo_projection"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Goles Esp. Favor (λ)", f"{swap['new_lambda']}", delta=f"{swap['delta_lambda']:+0.3f}")
            m2.metric("Goles Esp. Contra (μ)", f"{swap['new_mu']}", delta=f"{swap['delta_mu']:+0.3f}", delta_color="inverse")
            m3.metric("Probabilidad Victoria", f"{swap['new_win_prob']*100:.1f}%", delta=f"{swap['win_prob_delta']*100:+0.1f}%")
            m4.metric("Puntos Proyectados (38j)", f"{mc['new_projected_points_mean']} pts", delta=f"{mc['points_gain_mean']:+0.2f} pts")

            c_mc1, c_mc2 = st.columns(2)
            with c_mc1:
                st.subheader("Impacto en Clasificación Europea (>= 60 pts)")
                st.progress(mc['new_prob_europe_60pts'])
                st.write(f"Probabilidad con Fichaje: **{mc['new_prob_europe_60pts']*100:.1f}%** (Base: {mc['base_prob_europe_60pts']*100:.1f}%)")
            with c_mc2:
                st.subheader("Riesgo de Descenso (<= 38 pts)")
                st.progress(mc['new_prob_relegation_38pts'])
                st.write(f"Probabilidad con Fichaje: **{mc['new_prob_relegation_38pts']*100:.1f}%** (Base: {mc['base_prob_relegation_38pts']*100:.1f}%)")

    # =========================================================================
    # PESTAÑA 3: Preparación de Partido
    # =========================================================================
    elif menu == "3. Preparación de Partido (Matchup vs Rival)":
        st.title("⚔️ Preparación de Partido y Pizarra Táctica")
        st.markdown("Cruza las métricas de tu plantilla con las debilidades del rival para recomendar el Once Óptimo y jugadas clave.")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            my_team = st.selectbox("Tu Equipo:", sorted(df_all[df_all["league"] == "LaLiga"]["team_name"].unique()), index=0)
        with col_m2:
            rivals = [t for t in sorted(df_all[df_all["league"] == "LaLiga"]["team_name"].unique()) if t != my_team]
            rival_team = st.selectbox("Próximo Rival:", rivals, index=0)

        # Perfil táctico del rival
        st.subheader("Configuración del Perfil del Rival")
        pr1, pr2, pr3 = st.columns(3)
        with pr1:
            def_weak = st.slider("Debilidad Defensiva Rival (β):", 0.5, 2.0, 1.25)
        with pr2:
            att_threat = st.slider("Amenaza Ofensiva Rival (α):", 0.5, 2.0, 1.10)
        with pr3:
            high_line = st.slider("Vulnerabilidad Defensa Adelantada:", 0.0, 1.0, 0.70)

        opp_profile = {
            "defense_weakness": def_weak,
            "attack_threat": att_threat,
            "high_line_vulnerability": high_line,
            "low_block_tendency": 1.0 - high_line
        }

        plan = desk.prepare_match_plan(
            team_name=my_team,
            opponent_profile=opp_profile,
            formation={"GK": 1, "DF": 4, "MF": 3, "FW": 3}
        )

        st.subheader("🏆 Once Titular Recomendado vs " + rival_team)
        st.dataframe(plan["optimal_lineup"][["player_name", "position", "tactical_fit_score", "xg_p90", "xa_p90", "interceptions_p90"]], use_container_width=True)

        st.subheader("📌 Directrices Tácticas y Pizarra del Analista")
        for play in plan["tactical_plays"]:
            with st.expander(f"🔹 {play['categoria']}: {play['jugada']}", expanded=True):
                st.write(f"**Contexto Estadístico:** {play['contexto']}")
                st.info(f"**Instrucción al Cuerpo Técnico:** {play['instruccion_analista']}")
                st.success(f"**Impacto Esperado:** {play['impacto_esperado']}")

    # =========================================================================
    # PESTAÑA 4: Comparador de Perfiles (Radar p90)
    # =========================================================================
    elif menu == "4. Comparador de Perfiles (Radar p90)":
        st.title("📊 Comparador Contextual de Jugadores (Radar Chart)")
        
        c1, c2 = st.columns(2)
        with c1:
            p1_name = st.selectbox("Jugador A:", df_all["player_name"].unique(), index=0)
            p1 = df_all[df_all["player_name"] == p1_name].iloc[0]
        with c2:
            p2_name = st.selectbox("Jugador B:", df_all["player_name"].unique(), index=1)
            p2 = df_all[df_all["player_name"] == p2_name].iloc[0]

        categories = ["xG p90", "xA p90", "Intercepciones p90", "Tackles p90"]
        val1 = [p1["xg_p90"], p1["xa_p90"], p1["interceptions_p90"], p1["tackles_p90"]]
        val2 = [p2["xg_p90"], p2["xa_p90"], p2["interceptions_p90"], p2["tackles_p90"]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=val1 + [val1[0]], theta=categories + [categories[0]], fill='toself', name=f"{p1_name} ({p1['team_name']})"))
        fig.add_trace(go.Scatterpolar(r=val2 + [val2[0]], theta=categories + [categories[0]], fill='toself', name=f"{p2_name} ({p2['team_name']})"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
