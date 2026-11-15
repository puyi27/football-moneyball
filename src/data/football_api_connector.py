import os
import time
from typing import Dict, List, Any, Optional
import requests
import pandas as pd


class FootballAPIConnector:
    """
    Conector de APIs Deportivas en Vivo y Gestor de Fuentes de Datos Mundiales.
    Permite:
      1. Conexión con API-Football (API-Sports / RapidAPI) para extraer plantillas y estadísticas mundiales.
      2. Conexión con Football-Data.org para calendarios y alineaciones europeas.
      3. Búsqueda en vivo de jugadores por nombre o club.
      4. Normalización automática de métricas al esquema estándar de 'Football Moneyball AI'.
    """

    API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
    FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

    # Mapeo de códigos de competición para Football-Data.org
    COMPETITIONS_MAP = {
        "LaLiga": "PD",
        "Premier League": "PL",
        "Serie A": "SA",
        "Bundesliga": "BL1",
        "Ligue 1": "FL1",
        "Champions League": "CL",
        "Eredivisie": "DED",
        "Liga Portugal": "PPL",
        "Championship": "ELC"
    }

    def __init__(
        self,
        api_football_key: Optional[str] = None,
        football_data_token: Optional[str] = None,
        timeout: int = 8
    ):
        self.api_football_key = api_football_key or os.getenv("API_FOOTBALL_KEY", "")
        self.football_data_token = football_data_token or os.getenv("FOOTBALL_DATA_TOKEN", "")
        self.timeout = timeout

    def test_api_football_connection(self) -> Dict[str, Any]:
        """Verifica la conectividad y estado de la cuenta en API-Football."""
        if not self.api_football_key:
            return {
                "status": "NO_KEY",
                "message": "No se ha configurado la API Key de API-Football (RapidAPI / API-Sports).",
                "connected": False
            }

        headers = {
            "x-rapidapi-key": self.api_football_key,
            "x-apisports-key": self.api_football_key
        }

        try:
            start_time = time.time()
            resp = requests.get(f"{self.API_FOOTBALL_BASE_URL}/status", headers=headers, timeout=self.timeout)
            elapsed = round((time.time() - start_time) * 1000, 1)

            if resp.status_code == 200:
                data = resp.json()
                sub = data.get("response", {}).get("subscription", {})
                reqs = data.get("response", {}).get("requests", {})
                return {
                    "status": "SUCCESS",
                    "connected": True,
                    "latency_ms": elapsed,
                    "plan": sub.get("plan", "Free"),
                    "requests_remaining": reqs.get("limit_day", 100) - reqs.get("current", 0),
                    "message": f"Conexión activa con API-Football ({elapsed} ms). Peticiones restantes: {reqs.get('limit_day', 100) - reqs.get('current', 0)}"
                }
            else:
                return {
                    "status": f"HTTP_{resp.status_code}",
                    "connected": False,
                    "message": f"Error del servidor API-Football: {resp.status_code} - {resp.text}"
                }
        except Exception as e:
            return {
                "status": "CONNECTION_ERROR",
                "connected": False,
                "message": f"Fallo al conectar con API-Football: {str(e)}"
            }

    def test_football_data_connection(self) -> Dict[str, Any]:
        """Verifica la conectividad con Football-Data.org."""
        headers = {}
        if self.football_data_token:
            headers["X-Auth-Token"] = self.football_data_token

        try:
            start_time = time.time()
            resp = requests.get(f"{self.FOOTBALL_DATA_BASE_URL}/competitions/PD", headers=headers, timeout=self.timeout)
            elapsed = round((time.time() - start_time) * 1000, 1)

            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": "SUCCESS",
                    "connected": True,
                    "latency_ms": elapsed,
                    "competition": data.get("name"),
                    "message": f"Conectado a Football-Data.org ({elapsed} ms) para {data.get('name')}."
                }
            elif resp.status_code == 403:
                return {
                    "status": "AUTH_REQUIRED",
                    "connected": False,
                    "message": "Token requerido o límite alcanzado en Football-Data.org (código 403)."
                }
            else:
                return {
                    "status": f"HTTP_{resp.status_code}",
                    "connected": False,
                    "message": f"Respuesta HTTP {resp.status_code} de Football-Data.org."
                }
        except Exception as e:
            return {
                "status": "CONNECTION_ERROR",
                "connected": False,
                "message": f"Fallo al conectar con Football-Data.org: {str(e)}"
            }

    def search_live_player(self, player_name: str) -> List[Dict[str, Any]]:
        """
        Busca un jugador en vivo por nombre a través de API-Football si la clave está disponible.
        """
        if not self.api_football_key:
            return []

        headers = {
            "x-rapidapi-key": self.api_football_key,
            "x-apisports-key": self.api_football_key
        }
        params = {"search": player_name}

        try:
            resp = requests.get(f"{self.API_FOOTBALL_BASE_URL}/players", headers=headers, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                raw_players = data.get("response", [])
                results = []
                for item in raw_players:
                    p = item.get("player", {})
                    stats_list = item.get("statistics", [])
                    main_stat = stats_list[0] if stats_list else {}
                    team = main_stat.get("team", {}).get("name", "Sin equipo")
                    league = main_stat.get("league", {}).get("name", "Liga Desconocida")

                    results.append({
                        "player_id": p.get("id"),
                        "player_name": p.get("name"),
                        "team_name": team,
                        "league": league,
                        "position": self._map_position(main_stat.get("games", {}).get("position")),
                        "age": p.get("age", 25),
                        "nationality": p.get("nationality"),
                        "photo": p.get("photo"),
                    })
                return results
            else:
                return []
        except Exception:
            return []

    def fetch_competition_teams(self, league_name: str) -> List[Dict[str, Any]]:
        """
        Extrae los equipos de una liga desde Football-Data.org.
        """
        code = self.COMPETITIONS_MAP.get(league_name)
        if not code or not self.football_data_token:
            return []

        headers = {"X-Auth-Token": self.football_data_token}
        url = f"{self.FOOTBALL_DATA_BASE_URL}/competitions/{code}/teams"

        try:
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                teams = []
                for t in data.get("teams", []):
                    teams.append({
                        "team_id": t.get("id"),
                        "team_name": t.get("name"),
                        "short_name": t.get("shortName"),
                        "tla": t.get("tla"),
                        "crest": t.get("crest"),
                        "founded": t.get("founded"),
                        "venue": t.get("venue"),
                    })
                return teams
            return []
        except Exception:
            return []

    @staticmethod
    def _map_position(raw_pos: Optional[str]) -> str:
        """Normaliza la posición a la convención estándar (GK, DF, MF, FW)."""
        if not raw_pos:
            return "MF"
        pos = raw_pos.lower()
        if "goalkeeper" in pos or "portero" in pos or pos == "g":
            return "GK"
        elif "defender" in pos or "defensa" in pos or pos == "d":
            return "DF"
        elif "midfielder" in pos or "medio" in pos or "centrocampista" in pos or pos == "m":
            return "MF"
        elif "attacker" in pos or "delantero" in pos or "forward" in pos or pos == "f":
            return "FW"
        return "MF"

    def normalize_api_player_to_schema(self, raw_player: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convierte la respuesta en bruto de una API externa en el registro normalizado
        esperado por los modelos analíticos y la base de datos PostgreSQL.
        """
        pos = self._map_position(raw_player.get("position"))
        age = int(raw_player.get("age", 24) or 24)
        mv = float(raw_player.get("market_value_m", 15.0) or 15.0)

        # Si no trae xg / xa (común en planes free), sintetizar o extrapolar con base en goles y asistencias
        goals = float(raw_player.get("goals_p90", 0.25))
        assists = float(raw_player.get("assists_p90", 0.18))
        xg = float(raw_player.get("xg_p90", goals * 0.95))
        xa = float(raw_player.get("xa_p90", assists * 0.90))

        return {
            "player_id": int(raw_player.get("player_id", 9999)),
            "player_name": str(raw_player.get("player_name", "Jugador Desconocido")),
            "team_name": str(raw_player.get("team_name", "Agente Libre")),
            "league": str(raw_player.get("league", "Internacional")),
            "position": pos,
            "age": age,
            "market_value_m": mv,
            "minutes": int(raw_player.get("minutes", 1800) or 1800),
            "xg_p90": round(xg, 2),
            "xa_p90": round(xa, 2),
            "goals_p90": round(goals, 2),
            "assists_p90": round(assists, 2),
            "interceptions_p90": float(raw_player.get("interceptions_p90", 1.2)),
            "tackles_p90": float(raw_player.get("tackles_p90", 1.8)),
            "dribbles_p90": float(raw_player.get("dribbles_p90", 1.5)),
            "prog_passes_p90": float(raw_player.get("prog_passes_p90", 4.5)),
            "is_reliable": True,
            "potential_ovr": int(raw_player.get("potential_ovr", 85 if age <= 21 else 80)),
            "is_academy": bool(raw_player.get("is_academy", age <= 20)),
            "academy_origin": str(raw_player.get("academy_origin", f"Cantera {raw_player.get('team_name', '')}")),
        }
