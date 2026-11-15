import os
import sys
import psycopg2

def get_connection():
    """
    Devuelve una conexión activa a la base de datos PostgreSQL 16.
    Soporta variables de entorno y fallback ordenado.
    """
    host = os.getenv("DB_HOST", "fb_postgres")
    port = int(os.getenv("DB_PORT", "5432"))
    database = os.getenv("DB_NAME", "football_db")
    user = os.getenv("DB_USER", "moneyball")
    password = os.getenv("DB_PASSWORD", "scouting_admin")

    try:
        return psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
    except Exception as e_host:
        # Fallback a localhost en caso de ejecución fuera de la red docker de fb_postgres
        if host != "localhost":
            try:
                return psycopg2.connect(
                    host="localhost",
                    port=port,
                    database=database,
                    user=user,
                    password=password
                )
            except Exception:
                pass
        print(f"Error de conexión a PostgreSQL: {e_host}")
        raise e_host

def ensure_analytics_table(conn):
    """
    Garantiza la existencia de la tabla analytics_scouting_metrics
    y su índice único (player_id, season) para soportar UPSERTs idempotentes.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics_scouting_metrics (
            player_id BIGINT NOT NULL,
            season VARCHAR(20) NOT NULL,
            player_name VARCHAR(150),
            team_name VARCHAR(100),
            league VARCHAR(50),
            position VARCHAR(50),
            age BIGINT,
            market_value_m DOUBLE PRECISION,
            minutes BIGINT DEFAULT 0,
            goals_p90 DOUBLE PRECISION DEFAULT 0.0,
            assists_p90 DOUBLE PRECISION DEFAULT 0.0,
            xg_p90 DOUBLE PRECISION DEFAULT 0.0,
            xa_p90 DOUBLE PRECISION DEFAULT 0.0,
            interceptions_p90 DOUBLE PRECISION DEFAULT 0.0,
            tackles_p90 DOUBLE PRECISION DEFAULT 0.0,
            dribbles_p90 DOUBLE PRECISION DEFAULT 0.0,
            prog_passes_p90 DOUBLE PRECISION DEFAULT 0.0,
            goals_percentile DOUBLE PRECISION,
            assists_percentile DOUBLE PRECISION,
            xg_percentile DOUBLE PRECISION,
            xa_percentile DOUBLE PRECISION,
            is_reliable BOOLEAN DEFAULT FALSE,
            potential_ovr BIGINT,
            is_academy BOOLEAN DEFAULT FALSE,
            academy_origin VARCHAR(100),
            CONSTRAINT unique_scouting_player_season UNIQUE (player_id, season)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_analytics_scouting_metrics_player_season 
        ON analytics_scouting_metrics (player_id, season);

        ALTER TABLE analytics_scouting_metrics 
            ADD COLUMN IF NOT EXISTS goals_percentile DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS assists_percentile DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS xg_percentile DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS xa_percentile DOUBLE PRECISION;
        """)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Aviso asegurando analytics_scouting_metrics: {e}")
    finally:
        cursor.close()

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    schema_sql = """
    -- Dimensiones
    CREATE TABLE IF NOT EXISTS dim_teams (
        team_id SERIAL PRIMARY KEY,
        team_name VARCHAR(100) UNIQUE NOT NULL,
        league VARCHAR(50) NOT NULL
    );

    CREATE TABLE IF NOT EXISTS dim_players (
        player_id SERIAL PRIMARY KEY,
        player_name VARCHAR(150) NOT NULL,
        team_id INTEGER REFERENCES dim_teams(team_id) ON DELETE SET NULL,
        position VARCHAR(50),
        birth_year INTEGER,
        CONSTRAINT unique_player_team UNIQUE (player_name, team_id)
    );

    -- Hechos
    CREATE TABLE IF NOT EXISTS fact_matches (
        match_id SERIAL PRIMARY KEY,
        season VARCHAR(10) NOT NULL,
        match_date DATE NOT NULL,
        home_team_id INTEGER REFERENCES dim_teams(team_id) ON DELETE CASCADE,
        away_team_id INTEGER REFERENCES dim_teams(team_id) ON DELETE CASCADE,
        home_goals INTEGER NOT NULL,
        away_goals INTEGER NOT NULL,
        odds_home NUMERIC(5, 2),
        odds_draw NUMERIC(5, 2),
        odds_away NUMERIC(5, 2),
        CONSTRAINT unique_match UNIQUE (match_date, home_team_id, away_team_id)
    );

    CREATE TABLE IF NOT EXISTS fact_player_stats (
        stat_id SERIAL PRIMARY KEY,
        match_id INTEGER REFERENCES fact_matches(match_id) ON DELETE CASCADE,
        player_id INTEGER REFERENCES dim_players(player_id) ON DELETE CASCADE,
        minutes_played INTEGER NOT NULL DEFAULT 0,
        goals INTEGER DEFAULT 0,
        assists INTEGER DEFAULT 0,
        xg NUMERIC(4, 2) DEFAULT 0.0,
        xa NUMERIC(4, 2) DEFAULT 0.0,
        shots INTEGER DEFAULT 0,
        passes_completed INTEGER DEFAULT 0,
        tackles INTEGER DEFAULT 0,
        interceptions INTEGER DEFAULT 0,
        CONSTRAINT unique_player_match UNIQUE (match_id, player_id)
    );
    """

    try:
        cursor.execute(schema_sql)
        conn.commit()
        print("Esquema DDL de dimensiones y hechos creado exitosamente en PostgreSQL.")
    except Exception as e:
        conn.rollback()
        print(f"Error al crear el esquema: {e}")
        raise e
    finally:
        cursor.close()

    # Asegurar tabla analítica e índices de scouting
    ensure_analytics_table(conn)
    conn.close()

if __name__ == "__main__":
    init_db()