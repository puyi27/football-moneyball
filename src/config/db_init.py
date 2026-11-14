import psycopg2
import sys

def get_connection():
    try:
        return psycopg2.connect(
            host="fb_postgres",
            port="5432",
            database="football_db",
            user="moneyball",
            password="scouting_admin"
        )
    except Exception as e:
        print(f"Error de conexión a PostgreSQL: {e}")
        sys.exit(1)

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
        print("Esquema DDL creado exitosamente en PostgreSQL.")
    except Exception as e:
        conn.rollback()
        print(f"Error al crear el esquema: {e}")
        raise e
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    init_db()