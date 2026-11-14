import psycopg2
import pandas as pd
from src.config.db_init import get_connection

class DBLoader:
    """Clase encargada de la ingesta idempotente de datos en PostgreSQL."""

    def __init__(self):
        pass

    def get_db_connection(self):
        return get_connection()

    def insert_team_if_not_exists(self, conn, team_name, league="LaLiga"):
        """Inserta un equipo en dim_teams si no existe y devuelve su team_id."""
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dim_teams (team_name, league)
            VALUES (%s, %s)
            ON CONFLICT (team_name) DO UPDATE SET league = EXCLUDED.league
            RETURNING team_id;
        """, (team_name, league))
        team_id = cursor.fetchone()[0]
        cursor.close()
        return team_id

    def load_matches_df(self, df_matches, season="2526", league="LaLiga"):
        """Ingesta un DataFrame procesado de partidos en fact_matches."""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        inserted_count = 0

        try:
            for _, row in df_matches.iterrows():
                row_league = row['league'] if 'league' in row and pd.notnull(row['league']) else league
                home_team_id = self.insert_team_if_not_exists(conn, row['home_team'], row_league)
                away_team_id = self.insert_team_if_not_exists(conn, row['away_team'], row_league)

                match_date = row['match_date']
                home_goals = int(row['home_goals'])
                away_goals = int(row['away_goals'])
                
                odds_home = float(row['odds_home']) if 'odds_home' in row and pd.notnull(row['odds_home']) else None
                odds_draw = float(row['odds_draw']) if 'odds_draw' in row and pd.notnull(row['odds_draw']) else None
                odds_away = float(row['odds_away']) if 'odds_away' in row and pd.notnull(row['odds_away']) else None

                cursor.execute("""
                    INSERT INTO fact_matches (season, match_date, home_team_id, away_team_id, home_goals, away_goals, odds_home, odds_draw, odds_away)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (match_date, home_team_id, away_team_id) DO UPDATE SET
                        home_goals = EXCLUDED.home_goals,
                        away_goals = EXCLUDED.away_goals,
                        odds_home = EXCLUDED.odds_home,
                        odds_draw = EXCLUDED.odds_draw,
                        odds_away = EXCLUDED.odds_away;
                """, (season, match_date, home_team_id, away_team_id, home_goals, away_goals, odds_home, odds_draw, odds_away))
                inserted_count += 1

            conn.commit()
            print(f"Ingesta completada: {inserted_count} partidos procesados/actualizados.")
        except Exception as e:
            conn.rollback()
            print(f"Error cargando partidos en PostgreSQL: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

        return inserted_count

    def load_player_stats_df(self, df_players, league="LaLiga"):
        """Ingesta un DataFrame con métricas de jugadores en dim_players y fact_player_stats."""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        inserted_count = 0

        try:
            for _, row in df_players.iterrows():
                row_league = row['league'] if 'league' in row and pd.notnull(row['league']) else league
                team_name = row['team_name'] if pd.notnull(row['team_name']) else "Desconocido"
                team_id = self.insert_team_if_not_exists(conn, team_name, row_league)

                player_name = row['player_name']
                position = str(row['position']) if pd.notnull(row['position']) else None

                # Insertar o actualizar jugador en dim_players
                cursor.execute("""
                    INSERT INTO dim_players (player_name, team_id, position)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (player_name, team_id) DO UPDATE SET
                        position = EXCLUDED.position
                    RETURNING player_id;
                """, (player_name, team_id, position))
                player_id = cursor.fetchone()[0]
                inserted_count += 1

            conn.commit()
            print(f"Ingesta de dimensión jugadores completada: {inserted_count} registros.")
        except Exception as e:
            conn.rollback()
            print(f"Error cargando jugadores en PostgreSQL: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

        return inserted_count