import pytest
import psycopg2
from src.config.db_init import init_db, get_connection

def test_creacion_esquema():
    """Ejecuta el script DDL y verifica que las tablas existen en PostgreSQL."""
    init_db()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    tablas_esperadas = ["dim_teams", "dim_players", "fact_matches", "fact_player_stats"]
    
    for tabla in tablas_esperadas:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """, (tabla,))
        existe = cursor.fetchone()[0]
        assert existe is True, f"La tabla {tabla} no fue creada en la base de datos."
        
    cursor.close()
    conn.close()

def test_integridad_referencial():
    """Valida la restricción de clave foránea en dim_players."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Intentar insertar un jugador con un team_id inexistente (ej. 99999) debe fallar por FK
    try:
        cursor.execute("""
            INSERT INTO dim_players (player_name, team_id, position)
            VALUES ('Jugador Invalido', 99999, 'Delantero');
        """)
        conn.commit()
        pytest.fail("Se permitió insertar un jugador con un team_id inexistente.")
    except psycopg2.Error:
        conn.rollback()  # Comportamiento correcto
    finally:
        cursor.close()
        conn.close()