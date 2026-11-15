import os
import re
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import requests
import pandas as pd

logger = logging.getLogger("FootballAPIConnector")

# Directorio de caché local para evitar agotamiento de cuotas y bloqueos 429/403
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache_sources"
os.makedirs(CACHE_DIR / "api_football", exist_ok=True)
os.makedirs(CACHE_DIR / "football_data", exist_ok=True)


class FootballAPIConnector:
    """
    Conector de APIs Deportivas en Vivo y Gestor de Fuentes de Datos Mundiales.
    Permite:
      1. Conexión con API-Football (API-Sports / RapidAPI) para extraer plantillas y estadísticas mundiales.
      2. Conexión con Football-Data.org para calendarios, clasificaciones y alineaciones europeas.
      3. Búsqueda en vivo de jugadores por nombre o club con mitigación de rate limits (HTTP 429/403) y caché en disco.
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
        timeout: int = 8,
        cache_ttl_hours: int = 24
    ):
        self.api_football_key = api_football_key or os.getenv("API_FOOTBALL_KEY", "")
        self.football_data_token = football_data_token or os.getenv("FOOTBALL_DATA_TOKEN", "")
        self.timeout = timeout
        self.cache_ttl_hours = cache_ttl_hours

    # -------------------------------------------------------------------------
    # GESTIÓN DE CACHÉ EN DISCO & ESCUDO HTTP 429/403
    # -------------------------------------------------------------------------
    def _get_cache(self, subfolder: str, key: str) -> Optional[Any]:
        safe_key = hashlib.md5(key.encode("utf-8")).hexdigest()
        cache_path = CACHE_DIR / subfolder / f"{safe_key}.json"
        if not cache_path.exists():
            return None
        try:
            mtime = cache_path.stat().st_mtime
            if (time.time() - mtime) > (self.cache_ttl_hours * 3600):
                return None
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _set_cache(self, subfolder: str, key: str, data: Any) -> None:
        try:
            safe_key = hashlib.md5(key.encode("utf-8")).hexdigest()
            cache_path = CACHE_DIR / subfolder / f"{safe_key}.json"
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # DIAGNÓSTICO Y CONECTIVIDAD
    # -------------------------------------------------------------------------
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
            elif resp.status_code == 429:
                return {
                    "status": "HTTP_429",
                    "connected": False,
                    "message": "Límite de tasa (Rate Limit) alcanzado en API-Football. Operando con caché en disco y datalake local."
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
            elif resp.status_code in [401, 403]:
                return {
                    "status": "AUTH_REQUIRED",
                    "connected": False,
                    "message": "Token requerido o cuota consumida en Football-Data.org (código 403)."
                }
            elif resp.status_code == 429:
                return {
                    "status": "RATE_LIMITED",
                    "connected": False,
                    "message": "Demasiadas peticiones (HTTP 429) en Football-Data.org. Activando caché local."
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

    # -------------------------------------------------------------------------
    # BÚSQUEDA Y EXTRACCIÓN CON CACHÉ Y FALLBACK RESILIENTE
    # -------------------------------------------------------------------------
    def search_live_player(self, player_name: str, season: str = "2025/2026") -> List[Dict[str, Any]]:
        """
        Busca un jugador en vivo por nombre a través de API-Football con caché persistente
        y failover inmediato a la base de datos de contingencia en disco si no hay clave o hay error 429.
        """
        if not player_name or len(player_name.strip()) < 2:
            return []

        clean_name = player_name.strip()
        cache_key = f"search_{clean_name}_{season}"
        cached = self._get_cache("api_football", cache_key)
        if cached is not None:
            return cached

        if self.api_football_key:
            headers = {
                "x-rapidapi-key": self.api_football_key,
                "x-apisports-key": self.api_football_key
            }
            params = {"search": clean_name}
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
                            "season": season
                        })
                    if results:
                        self._set_cache("api_football", cache_key, results)
                        return results
            except Exception as e:
                logger.warning(f"Error consultando API-Football para '{clean_name}': {e}")

        # Fallback resiliente: buscar en la base de datos de futbolistas reales local
        local_results = self._search_local_dataset(clean_name, season)
        if local_results:
            self._set_cache("api_football", cache_key, local_results)
            return local_results

        return []

    def _search_local_dataset(self, name_query: str, season: str = "2025/2026") -> List[Dict[str, Any]]:
        """Búsqueda local resiliente en los archivos procesados del datalake."""
        db_file = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "global_players_database.csv"
        rp_file = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "real_players.csv"

        df_target = None
        if db_file.exists():
            try:
                df_target = pd.read_csv(db_file)
            except Exception:
                pass
        elif rp_file.exists():
            try:
                df_target = pd.read_csv(rp_file)
            except Exception:
                pass

        if df_target is None or df_target.empty:
            return []

        # Filtrado por subcadena insensible a mayúsculas
        q = name_query.lower()
        mask = df_target["player_name"].astype(str).str.lower().str.contains(q, na=False)
        matched = df_target[mask].head(10)

        results = []
        for _, row in matched.iterrows():
            results.append({
                "player_id": int(row.get("player_id", 9999)),
                "player_name": str(row.get("player_name", name_query)),
                "team_name": str(row.get("team_name", "Equipo")),
                "league": str(row.get("league", "LaLiga")),
                "position": self._map_position(str(row.get("position", "MF"))),
                "age": int(row.get("age", 24)),
                "market_value_m": float(row.get("market_value_m", 15.0)),
                "nationality": str(row.get("academy_origin", "España")),
                "photo": None,
                "season": season
            })
        return results

    def fetch_competition_teams(self, league_name: str, season: str = "2025/2026") -> List[Dict[str, Any]]:
        """
        Extrae los equipos de una liga desde Football-Data.org con caché local
        y fallback exhaustivo de clubes oficiales.
        """
        code = self.COMPETITIONS_MAP.get(league_name)
        cache_key = f"teams_{league_name}_{season}"
        cached = self._get_cache("football_data", cache_key)
        if cached is not None:
            return cached

        if code and self.football_data_token:
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
                            "league": league_name
                        })
                    if teams:
                        self._set_cache("football_data", cache_key, teams)
                        return teams
            except Exception as e:
                logger.warning(f"Error consultando equipos en Football-Data.org: {e}")

        # Fallback offline con clubes de Primera División de LaLiga
        if league_name == "LaLiga":
            teams = [
                {"team_id": 559, "team_name": "Sevilla FC", "short_name": "Sevilla", "tla": "SEV", "venue": "Ramón Sánchez-Pizjuán", "league": "LaLiga"},
                {"team_id": 86, "team_name": "Real Madrid CF", "short_name": "Real Madrid", "tla": "RMA", "venue": "Santiago Bernabéu", "league": "LaLiga"},
                {"team_id": 81, "team_name": "FC Barcelona", "short_name": "Barça", "tla": "FCB", "venue": "Spotify Camp Nou", "league": "LaLiga"},
                {"team_id": 78, "team_name": "Club Atlético de Madrid", "short_name": "Atleti", "tla": "ATM", "venue": "Riyadh Air Metropolitano", "league": "LaLiga"},
                {"team_id": 77, "team_name": "Athletic Club", "short_name": "Athletic", "tla": "ATH", "venue": "San Mamés", "league": "LaLiga"},
                {"team_id": 90, "team_name": "Real Betis Balompié", "short_name": "Betis", "tla": "BET", "venue": "Benito Villamarín", "league": "LaLiga"},
                {"team_id": 92, "team_name": "Real Sociedad de Fútbol", "short_name": "Real Sociedad", "tla": "RSO", "venue": "Reale Arena", "league": "LaLiga"},
                {"team_id": 94, "team_name": "Villarreal CF", "short_name": "Villarreal", "tla": "VIL", "venue": "Estadio de la Cerámica", "league": "LaLiga"},
                {"team_id": 95, "team_name": "Valencia CF", "short_name": "Valencia", "tla": "VAL", "venue": "Mestalla", "league": "LaLiga"},
                {"team_id": 298, "team_name": "Girona FC", "short_name": "Girona", "tla": "GIR", "venue": "Montilivi", "league": "LaLiga"},
                {"team_id": 87, "team_name": "Rayo Vallecano de Madrid", "short_name": "Rayo", "tla": "RAY", "venue": "Vallecas", "league": "LaLiga"},
                {"team_id": 79, "team_name": "CA Osasuna", "short_name": "Osasuna", "tla": "OSA", "venue": "El Sadar", "league": "LaLiga"},
                {"team_id": 89, "team_name": "RCD Mallorca", "short_name": "Mallorca", "tla": "MLL", "venue": "Son Moix", "league": "LaLiga"},
                {"team_id": 80, "team_name": "RCD Espanyol de Barcelona", "short_name": "Espanyol", "tla": "ESP", "venue": "Stage Front Stadium", "league": "LaLiga"},
                {"team_id": 263, "team_name": "Deportivo Alavés", "short_name": "Alavés", "tla": "ALA", "venue": "Mendizorroza", "league": "LaLiga"},
                {"team_id": 264, "team_name": "Getafe CF", "short_name": "Getafe", "tla": "GET", "venue": "Coliseum", "league": "LaLiga"},
                {"team_id": 275, "team_name": "UD Las Palmas", "short_name": "Las Palmas", "tla": "LPA", "venue": "Estadio Gran Canaria", "league": "LaLiga"},
                {"team_id": 267, "team_name": "RC Celta de Vigo", "short_name": "Celta", "tla": "CEL", "venue": "Abanca Balaídos", "league": "LaLiga"},
                {"team_id": 250, "team_name": "Real Valladolid CF", "short_name": "Valladolid", "tla": "VLD", "venue": "José Zorrilla", "league": "LaLiga"},
                {"team_id": 745, "team_name": "CD Leganés", "short_name": "Leganés", "tla": "LEG", "venue": "Butarque", "league": "LaLiga"}
            ]
            self._set_cache("football_data", cache_key, teams)
            return teams

        return []

    # -------------------------------------------------------------------------
    # NORMALIZACIÓN DE ESQUEMA CANÓNICO
    # -------------------------------------------------------------------------
    @staticmethod
    def _map_position(raw_pos: Optional[str]) -> str:
        """Normaliza la posición a la convención estándar (GK, DF, MF, FW)."""
        if not raw_pos:
            return "MF"
        pos = str(raw_pos).lower().strip()
        if any(term in pos for term in ["goalkeeper", "portero", "guardameta", "arquero"]) or pos == "g" or pos == "gk":
            return "GK"
        elif any(term in pos for term in ["defender", "defensa", "central", "lateral", "back", "cb", "lb", "rb"]) or pos == "d" or pos == "df":
            return "DF"
        elif any(term in pos for term in ["midfielder", "medio", "centrocampista", "pivote", "volante", "interior", "cm", "dm", "am"]) or pos == "m" or pos == "mf":
            return "MF"
        elif any(term in pos for term in ["attacker", "delantero", "forward", "extremo", "winger", "striker", "ariete", "cf", "rw", "lw"]) or pos == "f" or pos == "fw":
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

        # Extrapolar xg / xa con base en goles y asistencias o valores predeterminados
        goals = float(raw_player.get("goals_p90", 0.25) or 0.25)
        assists = float(raw_player.get("assists_p90", 0.18) or 0.18)
        xg = float(raw_player.get("xg_p90", goals * 0.95 if goals > 0 else 0.22))
        xa = float(raw_player.get("xa_p90", assists * 0.90 if assists > 0 else 0.16))

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

