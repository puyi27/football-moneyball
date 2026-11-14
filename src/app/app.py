import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import numpy as np

# 1. Configuración de página (debe ser la primera orden)
st.set_page_config(page_title="Football Moneyball AI", page_icon="⚽", layout="wide")

# 2. Conexión a BD cacheada
@st.cache_resource
def get_db_engine():
    # Nos conectamos usando el nombre del servicio de Docker 'postgres'
    return create_engine("postgresql://moneyball:scouting_admin@postgres:5432/football_db")

# 3. Extracción de datos cacheada
@st.cache_data
def get_scouting_data():
    try:
        engine = get_db_engine()
        # Aquí iría la query real a tu vista de scouting. 
        # Usamos un fallback robusto en caso de que la tabla aún esté vacía
        query = "SELECT * FROM players LIMIT 100" 
        df = pd.read_sql(query, engine)
    except Exception:
        # Fallback temporal para visualizar UI si la BD no tiene la tabla exacta
        np.random.seed(42)
        df = pd.DataFrame({
            "jugador": [f"Jugador {i}" for i in range(1, 101)],
            "edad": np.random.randint(18, 35, 100),
            "minutos": np.random.randint(450, 3000, 100),
            "goles_p90": np.random.uniform(0, 1.2, 100),
            "asistencias_p90": np.random.uniform(0, 0.8, 100),
            "valor_mercado_m": np.random.uniform(0.5, 50, 100)
        })
    return df

def main():
    st.sidebar.title("Football Moneyball AI")
    st.sidebar.markdown("---")
    
    modulo = st.sidebar.radio(
        "Selecciona el Módulo Operativo:",
        ("Módulo A: Scouting & K-NN", "Módulo B: Predicción & Poisson")
    )

    if modulo == "Módulo A: Scouting & K-NN":
        st.title("🔍 Radar de Scouting: Jugadores Infravalorados")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            edad_max = st.slider("Edad Máxima", 16, 40, 25)
        with col2:
            minutos_min = st.slider("Minutos Mínimos Jugados", 0, 3000, 450, step=90)
        with col3:
            metrica_y = st.selectbox("Métrica Eje Y", ["goles_p90", "asistencias_p90", "valor_mercado_m"])

        # Cargar y filtrar datos
        df = get_scouting_data()
        df_filtered = df[(df["edad"] <= edad_max) & (df["minutos"] >= minutos_min)]

        # Visualización Plotly
        st.subheader("Rendimiento Ofensivo vs Edad")
        if not df_filtered.empty:
            fig = px.scatter(
                df_filtered, 
                x="edad", 
                y=metrica_y, 
                hover_name="jugador",
                size="minutos",
                color="valor_mercado_m",
                color_continuous_scale="Viridis",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de datos
            st.dataframe(df_filtered.sort_values(metrica_y, ascending=False).head(10), use_container_width=True)
        else:
            st.warning("No hay jugadores que cumplan los filtros actuales.")

    elif modulo == "Módulo B: Predicción & Poisson":
        st.title("📈 Analítica de Partidos y Cuotas")
        st.info("Módulo en construcción. Aquí inyectaremos el modelo Elo y Dixon-Coles.")

if __name__ == "__main__":
    main()