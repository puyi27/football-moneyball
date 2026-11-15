import psycopg2
import pandas as pd
from src.config.db_init import get_connection, ensure_analytics_table

class DBLoader:
    """Clase encargada de la ingesta idempotente de datos en PostgreSQL 16."""

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

    def load_scouting_metrics_df(self, df_metrics, season="2026/2027"):
        """
        Ingesta idempotente (UPSERT) de métricas de scouting en PostgreSQL (analytics_scouting_metrics).
        Soporta DataFrames de pandas o PySpark.
        Garantiza idempotencia en PostgreSQL 16 mediante ON CONFLICT (player_id, season) DO UPDATE.
        """
        if hasattr(df_metrics, "toPandas"):
            df = df_metrics.toPandas()
        else:
            df = df_metrics.copy()

        if df.empty:
            print("DataFrame de métricas vacío. Ningún registro que procesar.")
            return 0

        conn = self.get_db_connection()
        ensure_analytics_table(conn)
        cursor = conn.cursor()

        try:
            # Normalización de nombres de columnas
            if "minutes_played" in df.columns and "minutes" not in df.columns:
                df["minutes"] = df["minutes_played"]
            if "season" not in df.columns:
                df["season"] = season
            else:
                df["season"] = df["season"].fillna(season)

            if "player_id" not in df.columns:
                raise ValueError("El DataFrame a cargar debe contener la columna 'player_id'.")

            # Obtener columnas existentes en analytics_scouting_metrics
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'analytics_scouting_metrics';
            """)
            db_cols = {row[0] for row in cursor.fetchall()}

            # Columnas del DataFrame válidas para inserción
            cols_to_insert = [c for c in df.columns if c in db_cols]
            if "player_id" not in cols_to_insert or "season" not in cols_to_insert:
                raise ValueError("Faltan columnas de clave única (player_id, season) para el UPSERT.")

            update_cols = [c for c in cols_to_insert if c not in ("player_id", "season")]
            cols_str = ", ".join(cols_to_insert)

            if update_cols:
                update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
                conflict_clause = f"DO UPDATE SET {update_clause}"
            else:
                conflict_clause = "DO NOTHING"

            upsert_sql = f"""
                INSERT INTO analytics_scouting_metrics ({cols_str})
                VALUES %s
                ON CONFLICT (player_id, season) {conflict_clause};
            """

            # Sanitizar valores None / NaN / tipos
            records = []
            for _, row in df[cols_to_insert].iterrows():
                record = tuple(None if pd.isna(val) else val for val in row)
                records.append(record)

            from psycopg2.extras import execute_values
            execute_values(cursor, upsert_sql, records, page_size=1000)
            conn.commit()

            count = len(records)
            print(f"UPSERT idempotente completado: {count} métricas cargadas en analytics_scouting_metrics.")
            return count
        except Exception as e:
            conn.rollback()
            print(f"Error cargando métricas en analytics_scouting_metrics: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

    upsert_scouting_metrics = load_scouting_metrics_df

    def upsert_from_staging(self, staging_table="stage_analytics_scouting_metrics", season="2026/2027"):
        """
        Ejecuta un UPSERT idempotente en PostgreSQL desde una tabla staging (p. ej. alimentada por PySpark JDBC)
        hacia analytics_scouting_metrics, fusionando sobre (player_id, season).
        """
        conn = self.get_db_connection()
        ensure_analytics_table(conn)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'analytics_scouting_metrics';
            """)
            dest_cols = {row[0] for row in cursor.fetchall()}

            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{staging_table}';
            """)
            stage_cols = {row[0] for row in cursor.fetchall()}

            if not stage_cols:
                print(f"Tabla de staging '{staging_table}' vacía o no existe.")
                return 0

            # Mapeo de columnas compatibles
            aliases = {"minutes": "minutes_played"}
            cols_to_insert = []
            select_exprs = []

            for col_name in dest_cols:
                if col_name in stage_cols:
                    cols_to_insert.append(col_name)
                    select_exprs.append(col_name)
                elif col_name in aliases and aliases[col_name] in stage_cols:
                    cols_to_insert.append(col_name)
                    select_exprs.append(f"{aliases[col_name]} AS {col_name}")

            if "season" not in cols_to_insert:
                cols_to_insert.append("season")
                select_exprs.append(f"'{season}' AS season")

            if "player_id" not in cols_to_insert:
                raise ValueError(f"No se encontró la columna 'player_id' en la tabla de staging '{staging_table}'.")

            update_cols = [c for c in cols_to_insert if c not in ("player_id", "season")]
            cols_str = ", ".join(cols_to_insert)
            select_str = ", ".join(select_exprs)

            if update_cols:
                update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
                conflict_clause = f"DO UPDATE SET {update_clause}"
            else:
                conflict_clause = "DO NOTHING"

            merge_sql = f"""
                INSERT INTO analytics_scouting_metrics ({cols_str})
                SELECT {select_str} FROM {staging_table}
                ON CONFLICT (player_id, season) {conflict_clause};
            """
            cursor.execute(merge_sql)
            merged_rows = cursor.rowcount
            conn.commit()
            print(f"Fusión idempotente completada: {merged_rows} registros integrados en analytics_scouting_metrics.")
            return merged_rows
        except Exception as e:
            conn.rollback()
            print(f"Error durante UPSERT desde staging ({staging_table}): {e}")
            raise e
        finally:
            cursor.close()
            conn.close()