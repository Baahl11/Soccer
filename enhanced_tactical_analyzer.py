"""
Enhanced Tactical Analyzer

Este módulo implementa un analizador táctico avanzado que puede generar perfiles tácticos
detallados para equipos de cualquier liga basados en sus estadísticas recientes y datos históricos.

Autor: Equipo de Desarrollo
Fecha: Mayo 25, 2025
"""

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import random
import numpy as np
from datetime import datetime
import json
import os

# Configuración de logging
logger = logging.getLogger(__name__)

class EnhancedTacticalAnalyzer:
    """
    Proporciona análisis tácticos detallados para cualquier equipo de fútbol basado en estadísticas recientes.
    Puede generar perfiles incluso cuando los datos son escasos, utilizando modelos aproximados.
    """
    
    def __init__(self, team_database_path: Optional[str] = None):
        """
        Inicializa el analizador táctico.
        
        Args:
            team_database_path: Ruta opcional al archivo de base de datos táctica de equipos
        """
        # Base de datos táctica por defecto
        self.default_team_database = {
            "possesion_styles": [
                {"name": "Posesión dominante", "description": "Mantienen el balón con pases cortos y buena circulación"},
                {"name": "Contraataque directo", "description": "Transiciones rápidas verticales después de recuperar"},
                {"name": "Presión alta", "description": "Recupera en campo rival con presión inmediata"},
                {"name": "Defensa baja + contraataques", "description": "Bloque bajo defensivo con salidas rápidas"},
                {"name": "Equilibrado", "description": "Adaptable según contexto del partido"}
            ],
            "defensive_styles": [
                {"name": "Presión alta", "description": "Presionan agresivamente en campo contrario"},
                {"name": "Bloque medio", "description": "Organizados en mediocampo con transiciones rápidas"},
                {"name": "Bloque bajo", "description": "Defensivamente sólidos cerca de su portería"},
                {"name": "Marca hombre a hombre", "description": "Asignaciones defensivas individuales"},
                {"name": "Defensa zonal", "description": "Cubren espacios en lugar de jugadores"}
            ],
            "offensive_styles": [
                {"name": "Ofensiva por bandas", "description": "Aprovechamiento de extremos para centros"},
                {"name": "Juego por el centro", "description": "Ataque por el medio con mediapuntas"},
                {"name": "Juego directo", "description": "Pases largos a delanteros"},
                {"name": "Ofensiva de posesión", "description": "Ataques elaborados con paciencia"},
                {"name": "Contragolpes veloces", "description": "Transición rápida defensa-ataque"}
            ],
            "formation_tendencies": [
                {"formation": "4-3-3", "style_tags": ["ofensivo", "bandas", "presión"]},
                {"formation": "4-4-2", "style_tags": ["equilibrado", "directo", "bloques"]},
                {"formation": "4-2-3-1", "style_tags": ["control", "mediapunta", "organizado"]},
                {"formation": "3-5-2", "style_tags": ["ofensivo", "carrileros", "contragolpes"]},
                {"formation": "5-3-2", "style_tags": ["defensivo", "contraataque", "sólido"]},
                {"formation": "3-4-3", "style_tags": ["ofensivo", "presión", "posesión"]}
            ],
            "set_piece_profiles": [
                {"name": "Especialistas balón parado", "description": "Equipo con buenos lanzadores y rematadores"},
                {"name": "Corners elaborados", "description": "Jugadas ensayadas en saques de esquina"},
                {"name": "Faltas directas", "description": "Especialistas en disparos directos"},
                {"name": "Defensa sólida a balón parado", "description": "Buen posicionamiento defensivo en córners"},
                {"name": "Juego aéreo fuerte", "description": "Aprovechan su altura en jugadas aéreas"}
            ],
            "teams": {}
        }
        
        # Intentar cargar base de datos de equipos si se proporciona ruta
        self.tactical_db = self.default_team_database
        if team_database_path:
            self.load_team_database(team_database_path)
    
    def load_team_database(self, database_path: str) -> bool:
        """
        Carga la base de datos táctica desde un archivo JSON.
        
        Args:
            database_path: Ruta al archivo JSON con datos tácticos
            
        Returns:
            True si la carga fue exitosa, False en caso contrario
        """
        try:
            if os.path.exists(database_path):
                with open(database_path, 'r', encoding='utf-8') as file:
                    loaded_db = json.load(file)
                    
                    # Validar estructura básica
                    required_keys = ["possesion_styles", "defensive_styles", 
                                    "offensive_styles", "teams"]
                    
                    if all(key in loaded_db for key in required_keys):
                        self.tactical_db = loaded_db
                        logger.info(f"Base de datos táctica cargada: {len(self.tactical_db.get('teams', {}))} equipos")
                        return True
                    else:
                        logger.warning(f"Formato inválido en base de datos táctica: {database_path}")
                        return False
            else:
                logger.warning(f"Archivo de base de datos táctica no encontrado: {database_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error cargando base de datos táctica: {str(e)}")
            return False
    
    def save_team_database(self, database_path: str) -> bool:
        """
        Guarda la base de datos táctica en un archivo JSON.
        
        Args:
            database_path: Ruta donde guardar el archivo JSON
            
        Returns:
            True si la operación fue exitosa, False en caso contrario
        """
        try:
            os.makedirs(os.path.dirname(database_path), exist_ok=True)
            
            with open(database_path, 'w', encoding='utf-8') as file:
                json.dump(self.tactical_db, file, indent=2, ensure_ascii=False)
                
            logger.info(f"Base de datos táctica guardada: {database_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando base de datos táctica: {str(e)}")
            return False
    
    def get_team_tactical_profile(self, 
                                team_id: int, 
                                team_name: str,
                                recent_formations: Optional[List[str]] = None,
                                recent_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Obtiene un perfil táctico detallado para un equipo, sea de la base de datos o generado.
        
        Args:
            team_id: ID del equipo
            team_name: Nombre del equipo
            recent_formations: Formaciones recientes utilizadas por el equipo
            recent_stats: Estadísticas recientes del equipo
            
        Returns:
            Un diccionario con el perfil táctico detallado del equipo
        """
        # Intentar buscar en la base de datos primero
        team_key = str(team_id)
        if team_key in self.tactical_db.get("teams", {}):
            # Verificar si el perfil está actualizado
            stored_profile = self.tactical_db["teams"][team_key]
            last_updated = datetime.fromisoformat(stored_profile.get("last_updated", "2000-01-01"))
            
            # Si ha pasado menos de un mes, usar perfil almacenado
            days_since_update = (datetime.now() - last_updated).days
            if days_since_update < 30:
                logger.info(f"Usando perfil táctico existente para {team_name} (ID: {team_id})")
                return stored_profile
        
        # Si no existe o está desactualizado, generar nuevo perfil
        logger.info(f"Generando nuevo perfil táctico para {team_name} (ID: {team_id})")
        
        # Generar perfil basado en datos o aproximado si no hay datos suficientes
        if recent_formations and recent_stats:
            profile = self._generate_data_based_profile(team_id, team_name, recent_formations, recent_stats)
        else:
            profile = self._generate_approximated_profile(team_id, team_name)
        
        # Guardar en la base de datos
        if "teams" not in self.tactical_db:
            self.tactical_db["teams"] = {}
        
        self.tactical_db["teams"][team_key] = profile
        
        return profile
    
    def _generate_data_based_profile(self,
                                   team_id: int,
                                   team_name: str,
                                   recent_formations: List[str],
                                   recent_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera un perfil táctico basado en datos reales del equipo.
        
        Args:
            team_id: ID del equipo
            team_name: Nombre del equipo
            recent_formations: Formaciones recientes utilizadas por el equipo
            recent_stats: Estadísticas recientes del equipo
            
        Returns:
            Un diccionario con el perfil táctico detallado
        """
        # Definir formato base del perfil
        profile = {
            "team_id": team_id,
            "team_name": team_name,
            "last_updated": datetime.now().isoformat(),
            "formations": [],
            "tactical_style": {},
            "strengths": [],
            "weaknesses": [],
            "set_pieces": {},
            "adaptability": 0.0
        }
        
        # Analizar formaciones recientes
        if recent_formations:
            # Contar frecuencia de cada formación
            formation_counts = {}
            for formation in recent_formations:
                if formation in formation_counts:
                    formation_counts[formation] += 1
                else:
                    formation_counts[formation] = 1
                    
            # Calcular porcentajes
            total_matches = len(recent_formations)
            formation_percentages = {f: (count / total_matches * 100) 
                                    for f, count in formation_counts.items()}
            
            # Ordenar por frecuencia
            sorted_formations = sorted(formation_percentages.items(), 
                                      key=lambda x: x[1], reverse=True)
            
            # Añadir al perfil
            profile["formations"] = [{"formation": f, "usage_percentage": p} 
                                    for f, p in sorted_formations]
            
            # Identificar formación principal
            main_formation = sorted_formations[0][0] if sorted_formations else "4-4-2"
        else:
            main_formation = "4-4-2"
            profile["formations"].append({"formation": main_formation, "usage_percentage": 100})
        
        # Determinar estilo táctico basado en estadísticas
        if recent_stats:
            # Posesión media
            avg_possession = recent_stats.get("avg_possession", 50.0)
            
            # Estilo de posesión
            if avg_possession > 55:
                profile["tactical_style"]["possession"] = self._get_random_style_from_db("possesion_styles", ["Posesión dominante", "Ofensiva de posesión"])
            elif avg_possession < 45:
                profile["tactical_style"]["possession"] = self._get_random_style_from_db("possesion_styles", ["Contraataque directo", "Defensa baja + contraataques"])
            else:
                profile["tactical_style"]["possession"] = self._get_random_style_from_db("possesion_styles", ["Equilibrado"])
            
            # Estilo ofensivo basado en goles y disparos
            goals_per_match = recent_stats.get("avg_goals_scored", 1.3)
            shots_per_match = recent_stats.get("avg_shots", 12.0)
            
            if goals_per_match > 2.0:
                if avg_possession > 52:
                    profile["tactical_style"]["offensive"] = self._get_random_style_from_db("offensive_styles", ["Ofensiva de posesión", "Juego por el centro"])
                else:
                    profile["tactical_style"]["offensive"] = self._get_random_style_from_db("offensive_styles", ["Contragolpes veloces", "Juego directo"])
            elif shots_per_match > 15:
                profile["tactical_style"]["offensive"] = self._get_random_style_from_db("offensive_styles", ["Ofensiva por bandas"])
            else:
                profile["tactical_style"]["offensive"] = self._get_random_style_from_db("offensive_styles")
                
            # Estilo defensivo
            goals_conceded = recent_stats.get("avg_goals_conceded", 1.2)
            tackles_per_match = recent_stats.get("avg_tackles", 18.0)
            
            if goals_conceded < 1.0:
                profile["tactical_style"]["defensive"] = self._get_random_style_from_db("defensive_styles", ["Bloque bajo", "Defensa zonal"])
            elif tackles_per_match > 22:
                profile["tactical_style"]["defensive"] = self._get_random_style_from_db("defensive_styles", ["Presión alta", "Marca hombre a hombre"])
            else:
                profile["tactical_style"]["defensive"] = self._get_random_style_from_db("defensive_styles", ["Bloque medio"])
            
            # Balón parado
            set_piece_goals = recent_stats.get("set_piece_goals", 0)
            corners_per_match = recent_stats.get("avg_corners", 5.0)
            
            if set_piece_goals > 5 or corners_per_match > 7:
                profile["set_pieces"] = self._get_random_style_from_db("set_piece_profiles", ["Especialistas balón parado", "Corners elaborados"])
            else:
                profile["set_pieces"] = self._get_random_style_from_db("set_piece_profiles")
                
            # Determinar fortalezas y debilidades
            strengths = []
            weaknesses = []
            
            # Analizar estadísticas para determinar fortalezas
            if avg_possession > 55:
                strengths.append("Control del partido")
            if goals_per_match > 2.0:
                strengths.append("Eficacia goleadora")
            if goals_conceded < 1.0:
                strengths.append("Solidez defensiva")
            if recent_stats.get("avg_cards", 2.0) < 1.5:
                strengths.append("Disciplina táctica")
            if corners_per_match > 6:
                strengths.append("Juego ofensivo por bandas")
                
            # Analizar estadísticas para determinar debilidades
            if avg_possession < 45:
                weaknesses.append("Dificultad para mantener posesión")
            if goals_per_match < 1.0:
                weaknesses.append("Falta de efectividad ofensiva")
            if goals_conceded > 1.8:
                weaknesses.append("Vulnerabilidad defensiva")
            if recent_stats.get("avg_cards", 2.0) > 3.0:
                weaknesses.append("Problemas disciplinarios")
                
            # Asegurar al menos 2 fortalezas y 2 debilidades
            while len(strengths) < 2:
                random_strength = random.choice([
                    "Juego en transición", "Intensidad física",
                    "Organización defensiva", "Juego combinativo"
                ])
                if random_strength not in strengths:
                    strengths.append(random_strength)
                    
            while len(weaknesses) < 2:
                random_weakness = random.choice([
                    "Defensa de contraataques", "Finalización de ocasiones",
                    "Juego aéreo defensivo", "Presión tras pérdida"
                ])
                if random_weakness not in weaknesses:
                    weaknesses.append(random_weakness)
            
            # Añadir al perfil
            profile["strengths"] = strengths[:4]  # Máximo 4
            profile["weaknesses"] = weaknesses[:3]  # Máximo 3
            
            # Determinar adaptabilidad táctica basada en variación de formaciones
            unique_formations = len(formation_counts) if formation_counts else 1
            adaptability = min(1.0, unique_formations / 4)
            profile["adaptability"] = round(adaptability, 2)
            
        else:
            # Si no hay estadísticas, generar un perfil aproximado
            return self._generate_approximated_profile(team_id, team_name, main_formation)
        
        return profile
    
    def _generate_approximated_profile(self, 
                                     team_id: int, 
                                     team_name: str,
                                     main_formation: Optional[str] = None) -> Dict[str, Any]:
        """
        Genera un perfil táctico aproximado cuando no hay datos suficientes.
        
        Args:
            team_id: ID del equipo
            team_name: Nombre del equipo
            main_formation: Formación principal del equipo (opcional)
            
        Returns:
            Un diccionario con el perfil táctico aproximado
        """
        # Definir formato base del perfil
        profile = {
            "team_id": team_id,
            "team_name": team_name,
            "last_updated": datetime.now().isoformat(),
            "formations": [],
            "tactical_style": {},
            "strengths": [],
            "weaknesses": [],
            "set_pieces": {},
            "adaptability": round(random.uniform(0.3, 0.7), 2),
            "approximated": True
        }
        
        # Seleccionar una formación principal si no se proporciona
        if not main_formation:
            main_formation = random.choice(["4-4-2", "4-3-3", "4-2-3-1", "3-5-2", "5-3-2"])
        
        # Añadir formaciones con porcentajes aproximados
        secondary_formation = random.choice([f["formation"] for f in self.tactical_db["formation_tendencies"] 
                                           if f["formation"] != main_formation])
        
        profile["formations"] = [
            {"formation": main_formation, "usage_percentage": random.randint(65, 85)},
            {"formation": secondary_formation, "usage_percentage": random.randint(15, 35)}
        ]
        
        # Normalizar porcentajes para que sumen 100
        total = sum(f["usage_percentage"] for f in profile["formations"])
        for f in profile["formations"]:
            f["usage_percentage"] = round((f["usage_percentage"] / total) * 100, 1)
        
        # Seleccionar estilos tácticos aleatorios pero coherentes
        profile["tactical_style"]["possession"] = self._get_random_style_from_db("possesion_styles")
        profile["tactical_style"]["offensive"] = self._get_random_style_from_db("offensive_styles")
        profile["tactical_style"]["defensive"] = self._get_random_style_from_db("defensive_styles")
        profile["set_pieces"] = self._get_random_style_from_db("set_piece_profiles")
        
        # Generar fortalezas y debilidades aleatorias pero coherentes
        all_strengths = [
            "Juego en transición", "Pressing organizado",
            "Sólida defensa", "Eficacia en ataque",
            "Juego aéreo", "Posesión de balón",
            "Contraataques", "Experiencia táctica",
            "Balón parado", "Consistencia defensiva"
        ]
        
        all_weaknesses = [
            "Defensa de contraataques", "Finalización de ocasiones",
            "Juego aéreo defensivo", "Presión tras pérdida",
            "Vulnerabilidad a balones largos", "Ritmo defensivo",
            "Creación de ocasiones", "Concentración en finales ajustados",
            "Profundidad ofensiva", "Adaptación táctica"
        ]
        
        # Seleccionar 3 fortalezas y 2 debilidades aleatorias
        profile["strengths"] = random.sample(all_strengths, 3)
        profile["weaknesses"] = random.sample(all_weaknesses, 2)
        
        return profile
    
    def _get_random_style_from_db(self, style_category: str, preferred_names: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Obtiene un estilo aleatorio de la base de datos táctica.
        
        Args:
            style_category: Categoría del estilo ('possession_styles', 'defensive_styles', etc.)
            preferred_names: Lista opcional de nombres de estilos preferidos
            
        Returns:
            Un diccionario con el estilo seleccionado
        """
        if style_category in self.tactical_db:
            styles = self.tactical_db[style_category]
            
            # Si hay nombres preferidos, filtrar por ellos
            if preferred_names:
                preferred_styles = [s for s in styles if s.get("name") in preferred_names]
                if preferred_styles:
                    return random.choice(preferred_styles)
            
            # Si no hay preferidos o no se encontraron, devolver cualquiera
            if styles:
                return random.choice(styles)
        
        # Si la categoría no existe o está vacía, devolver un estilo genérico
        return {"name": "Equilibrado", "description": "Estilo balanceado en todos los aspectos"}
    
    def generate_match_tactical_analysis(self,
                                       home_team_id: int,
                                       away_team_id: int,
                                       home_team_name: str,
                                       away_team_name: str,
                                       home_team_stats: Optional[Dict[str, Any]] = None,
                                       away_team_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Genera un análisis táctico detallado para un partido específico.
        
        Args:
            home_team_id: ID del equipo local
            away_team_id: ID del equipo visitante
            home_team_name: Nombre del equipo local
            away_team_name: Nombre del equipo visitante
            home_team_stats: Estadísticas recientes del equipo local (opcional)
            away_team_stats: Estadísticas recientes del equipo visitante (opcional)
            
        Returns:
            Un diccionario con análisis táctico detallado del partido
        """
        # Obtener perfiles tácticos de ambos equipos
        home_profile = self.get_team_tactical_profile(
            home_team_id, home_team_name, 
            recent_stats=home_team_stats
        )
        
        away_profile = self.get_team_tactical_profile(
            away_team_id, away_team_name, 
            recent_stats=away_team_stats
        )
        
        # Determinar formaciones previstas
        expected_home_formation = home_profile.get("formations", [])[0].get("formation", "4-4-2") if home_profile.get("formations") else "4-4-2"
        expected_away_formation = away_profile.get("formations", [])[0].get("formation", "4-4-2") if away_profile.get("formations") else "4-4-2"
        
        # Analizar choque de estilos
        home_style = home_profile.get("tactical_style", {})
        away_style = away_profile.get("tactical_style", {})
        
        # Predecir quién dominará la posesión
        home_possession_style = home_style.get("possession", {}).get("name", "Equilibrado")
        away_possession_style = away_style.get("possession", {}).get("name", "Equilibrado")
        
        possession_advantage = self._determine_possession_advantage(
            home_possession_style, away_possession_style,
            home_team_stats, away_team_stats
        )
        
        # Determinar áreas clave del partido
        key_areas = self._determine_key_tactical_areas(home_profile, away_profile)
        
        # Generar recomendaciones tácticas
        home_recommendations = self._generate_tactical_recommendations(home_profile, away_profile, True)
        away_recommendations = self._generate_tactical_recommendations(away_profile, home_profile, False)
        
        # Determinar puntos de enfoque
        focus_points = self._determine_match_focus_points(home_profile, away_profile)
        
        # Componer análisis táctico completo
        tactical_analysis = {
            "home_team": {
                "id": home_team_id,
                "name": home_team_name,
                "expected_formation": expected_home_formation,
                "tactical_approach": home_style,
                "strengths_to_exploit": home_profile.get("strengths", [])[:2],
                "recommendations": home_recommendations
            },
            "away_team": {
                "id": away_team_id,
                "name": away_team_name,
                "expected_formation": expected_away_formation,
                "tactical_approach": away_style,
                "strengths_to_exploit": away_profile.get("strengths", [])[:2],
                "recommendations": away_recommendations
            },
            "match_dynamics": {
                "possession_advantage": possession_advantage,
                "key_tactical_areas": key_areas,
                "focus_points": focus_points
            }
        }
        
        return tactical_analysis
    
    def _determine_possession_advantage(self,
                                      home_style: str,
                                      away_style: str,
                                      home_stats: Optional[Dict[str, Any]] = None,
                                      away_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Determina qué equipo tendrá ventaja en la posesión del balón.
        
        Args:
            home_style: Estilo de posesión del equipo local
            away_style: Estilo de posesión del equipo visitante
            home_stats: Estadísticas del equipo local (opcional)
            away_stats: Estadísticas del equipo visitante (opcional)
            
        Returns:
            Un diccionario con la ventaja de posesión
        """
        # Si hay estadísticas disponibles, usarlas como base
        if home_stats and away_stats:
            home_avg_possession = home_stats.get("avg_possession", 50.0)
            away_avg_possession = away_stats.get("avg_possession", 50.0)
            
            # Ajustar por ventaja local (2-3% extra de posesión)
            home_avg_possession += 2.5
            
            # Determinar quién tiene la ventaja
            if home_avg_possession > away_avg_possession + 5:
                advantage = "home"
                magnitude = "significativa" if home_avg_possession > away_avg_possession + 10 else "ligera"
                expected_possession = round(min(65, home_avg_possession), 1)
            elif away_avg_possession > home_avg_possession + 5:
                advantage = "away"
                magnitude = "significativa" if away_avg_possession > home_avg_possession + 10 else "ligera"
                expected_possession = round(min(65, away_avg_possession), 1)
            else:
                advantage = "neutral"
                magnitude = "equilibrada"
                expected_possession = round((home_avg_possession + away_avg_possession) / 2, 1)
        else:
            # Si no hay estadísticas, inferir del estilo de juego
            possession_weights = {
                "Posesión dominante": 3,
                "Ofensiva de posesión": 2,
                "Equilibrado": 0,
                "Contraataque directo": -2,
                "Defensa baja + contraataques": -3
            }
            
            home_weight = possession_weights.get(home_style, 0) + 1  # +1 por ventaja local
            away_weight = possession_weights.get(away_style, 0)
            
            if home_weight > away_weight + 1:
                advantage = "home"
                magnitude = "significativa" if home_weight > away_weight + 3 else "ligera"
                expected_possession = 55 if magnitude == "ligera" else 60
            elif away_weight > home_weight + 1:
                advantage = "away"
                magnitude = "significativa" if away_weight > home_weight + 3 else "ligera"
                expected_possession = 55 if magnitude == "ligera" else 60
            else:
                advantage = "neutral"
                magnitude = "equilibrada"
                expected_possession = 50
        
        return {
            "team": advantage,
            "magnitude": magnitude,
            "expected_possession_percentage": expected_possession,
            "description": self._get_possession_advantage_description(advantage, magnitude)
        }
    
    def _get_possession_advantage_description(self, advantage: str, magnitude: str) -> str:
        """
        Genera una descripción de la ventaja en posesión.
        
        Args:
            advantage: Equipo con ventaja ('home', 'away' o 'neutral')
            magnitude: Magnitud de la ventaja ('ligera', 'significativa' o 'equilibrada')
            
        Returns:
            Descripción de la ventaja de posesión
        """
        if advantage == "home":
            if magnitude == "significativa":
                return "El equipo local probablemente dominará claramente la posesión"
            else:
                return "El equipo local debería tener una ligera ventaja en posesión"
        elif advantage == "away":
            if magnitude == "significativa":
                return "El equipo visitante podría dominar la posesión a pesar de jugar fuera"
            else:
                return "El equipo visitante debería conseguir una ligera ventaja en posesión"
        else:
            return "Se espera una posesión equilibrada entre ambos equipos"
    
    def _determine_key_tactical_areas(self, home_profile: Dict[str, Any], away_profile: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Determina las áreas tácticas clave del partido.
        
        Args:
            home_profile: Perfil táctico del equipo local
            away_profile: Perfil táctico del equipo visitante
            
        Returns:
            Lista de áreas tácticas clave
        """
        key_areas = []
        
        # Comparar fortalezas vs debilidades
        home_strengths = home_profile.get("strengths", [])
        away_strengths = away_profile.get("strengths", [])
        home_weaknesses = home_profile.get("weaknesses", [])
        away_weaknesses = away_profile.get("weaknesses", [])
        
        # Buscar coincidencias entre fortalezas y debilidades
        for strength in home_strengths:
            for weakness in away_weaknesses:
                if self._are_concepts_related(strength, weakness):
                    key_areas.append({
                        "area": strength,
                        "advantage": "home",
                        "description": f"Ventaja local en {strength.lower()} vs {weakness.lower()} visitante"
                    })
        
        for strength in away_strengths:
            for weakness in home_weaknesses:
                if self._are_concepts_related(strength, weakness):
                    key_areas.append({
                        "area": strength,
                        "advantage": "away",
                        "description": f"Ventaja visitante en {strength.lower()} vs {weakness.lower()} local"
                    })
        
        # Si no se encuentran coincidencias, añadir áreas generales
        if not key_areas:
            # Comparar estilos de juego
            home_style = home_profile.get("tactical_style", {})
            away_style = away_profile.get("tactical_style", {})
            
            # Comparar estilos ofensivos vs defensivos
            if "offensive" in home_style and "defensive" in away_style:
                key_areas.append({
                    "area": "Ataque local vs defensa visitante",
                    "advantage": "neutral",
                    "description": f"{home_style['offensive']['name']} contra {away_style['defensive']['name']}"
                })
                
            if "offensive" in away_style and "defensive" in home_style:
                key_areas.append({
                    "area": "Ataque visitante vs defensa local",
                    "advantage": "neutral",
                    "description": f"{away_style['offensive']['name']} contra {home_style['defensive']['name']}"
                })
        
        return key_areas[:3]  # Limitar a 3 áreas clave
    
    def _are_concepts_related(self, concept1: str, concept2: str) -> bool:
        """
        Determina si dos conceptos tácticos están relacionados.
        
        Args:
            concept1: Primer concepto táctico
            concept2: Segundo concepto táctico
            
        Returns:
            True si los conceptos están relacionados, False en caso contrario
        """
        # Convertir a minúsculas para comparación
        c1 = concept1.lower()
        c2 = concept2.lower()
        
        # Palabras clave para diferentes aspectos tácticos
        defensive_keywords = ['defens', 'bloque', 'marca', 'presión', 'solidez', 'seguridad']
        offensive_keywords = ['ataque', 'goleador', 'ofensiv', 'finalizaci', 'ocasiones']
        possession_keywords = ['posesión', 'control', 'circulaci', 'pases']
        transition_keywords = ['transici', 'contraata', 'cambio']
        
        # Verificar si los conceptos pertenecen a la misma categoría
        def has_keywords(concept, keywords):
            return any(kw in concept for kw in keywords)
        
        # Dos conceptos están relacionados si pertenecen a la misma categoría
        for keywords in [defensive_keywords, offensive_keywords, possession_keywords, transition_keywords]:
            if has_keywords(c1, keywords) and has_keywords(c2, keywords):
                return True
                
        # O si uno es de ataque y otro de defensa (conceptos opuestos)
        if (has_keywords(c1, offensive_keywords) and has_keywords(c2, defensive_keywords)) or \
           (has_keywords(c1, defensive_keywords) and has_keywords(c2, offensive_keywords)):
            return True
            
        return False
    
    def _generate_tactical_recommendations(self, team_profile: Dict[str, Any], 
                                         opponent_profile: Dict[str, Any], 
                                         is_home: bool) -> List[str]:
        """
        Genera recomendaciones tácticas basadas en perfiles de equipos.
        
        Args:
            team_profile: Perfil del equipo para el que se generan recomendaciones
            opponent_profile: Perfil del equipo rival
            is_home: True si el equipo juega como local, False si es visitante
            
        Returns:
            Lista de recomendaciones tácticas
        """
        recommendations = []
        
        # Analizar fortalezas del equipo vs debilidades del oponente
        team_strengths = team_profile.get("strengths", [])
        opponent_weaknesses = opponent_profile.get("weaknesses", [])
        
        # Buscar oportunidades de explotación
        for strength in team_strengths:
            for weakness in opponent_weaknesses:
                if self._are_concepts_related(strength, weakness):
                    if is_home:
                        recommendations.append(f"Aprovechar {strength.lower()} contra {weakness.lower()} del rival")
                    else:
                        recommendations.append(f"Explotar {strength.lower()} a pesar de jugar fuera")
        
        # Recomendaciones basadas en estilo de juego
        team_style = team_profile.get("tactical_style", {})
        opponent_style = opponent_profile.get("tactical_style", {})
        
        # Posesión
        if "possession" in team_style and "possession" in opponent_style:
            team_possession = team_style["possession"].get("name", "")
            opponent_possession = opponent_style["possession"].get("name", "")
            
            if "Posesión dominante" in team_possession and "Contraataque" in opponent_possession:
                recommendations.append("Circular el balón con paciencia pero evitar pérdidas en zonas peligrosas")
            elif "Contraataque" in team_possession and "Posesión dominante" in opponent_possession:
                recommendations.append("Presionar en bloque medio y buscar transiciones rápidas")
        
        # Aspectos ofensivos vs defensivos
        if "offensive" in team_style and "defensive" in opponent_style:
            team_offensive = team_style["offensive"].get("name", "")
            opponent_defensive = opponent_style["defensive"].get("name", "")
            
            if "bandas" in team_offensive.lower() and "Bloque bajo" in opponent_defensive:
                recommendations.append("Utilizar centros altos y segundo palo para superar el bloque bajo")
            elif "directo" in team_offensive.lower() and "Presión alta" in opponent_defensive:
                recommendations.append("Buscar pases largos directos para superar la presión rival")
        
        # Si no hay recomendaciones específicas, añadir algunas genéricas
        if not recommendations:
            if is_home:
                recommendations.append("Imponer ritmo alto desde el inicio aprovechando el factor local")
                recommendations.append("Buscar profundidad en bandas para generar superioridades")
            else:
                recommendations.append("Mantener bloque compacto en los primeros minutos")
                recommendations.append("Aprovechar contragolpes y buscar eficacia en las llegadas")
        
        # Balón parado
        set_pieces = team_profile.get("set_pieces", {}).get("name", "")
        if "Especialistas" in set_pieces or "Corners" in set_pieces:
            recommendations.append("Maximizar ocasiones a balón parado como fortaleza del equipo")
            
        # Limitar a 3 recomendaciones
        return recommendations[:3]
    
    def _determine_match_focus_points(self, home_profile: Dict[str, Any], away_profile: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Determina puntos de enfoque clave para el partido.
        
        Args:
            home_profile: Perfil táctico del equipo local
            away_profile: Perfil táctico del equipo visitante
            
        Returns:
            Lista de puntos de enfoque clave del partido
        """
        focus_points = []
        
        # Analizar formaciones
        home_formation = home_profile.get("formations", [])[0].get("formation", "4-4-2") if home_profile.get("formations") else "4-4-2"
        away_formation = away_profile.get("formations", [])[0].get("formation", "4-4-2") if away_profile.get("formations") else "4-4-2"
        
        # Comparar formaciones para identificar ventajas numéricas
        home_numbers = self._parse_formation(home_formation)
        away_numbers = self._parse_formation(away_formation)
        
        # Comparar líneas para identificar ventajas numéricas
        if len(home_numbers) >= 3 and len(away_numbers) >= 3:
            # Comparar mediocampo
            if home_numbers[1] > away_numbers[1]:
                focus_points.append({
                    "area": "Mediocampo",
                    "advantage": "home",
                    "description": f"Superioridad numérica en mediocampo local ({home_numbers[1]} vs {away_numbers[1]})"
                })
            elif away_numbers[1] > home_numbers[1]:
                focus_points.append({
                    "area": "Mediocampo",
                    "advantage": "away",
                    "description": f"Superioridad numérica en mediocampo visitante ({away_numbers[1]} vs {home_numbers[1]})"
                })
            
            # Comparar defensa vs ataque
            if home_numbers[0] < away_numbers[2]:
                focus_points.append({
                    "area": "Ataque visitante",
                    "advantage": "away",
                    "description": f"Superioridad numérica del ataque visitante vs defensa local ({away_numbers[2]} vs {home_numbers[0]})"
                })
            elif away_numbers[0] < home_numbers[2]:
                focus_points.append({
                    "area": "Ataque local",
                    "advantage": "home",
                    "description": f"Superioridad numérica del ataque local vs defensa visitante ({home_numbers[2]} vs {away_numbers[0]})"
                })
        
        # Añadir puntos sobre estilos de juego
        home_style = home_profile.get("tactical_style", {})
        away_style = away_profile.get("tactical_style", {})
        
        # Verificar si hay un aspecto de presión alta vs juego directo
        if home_style.get("defensive", {}).get("name", "") == "Presión alta" and \
           away_style.get("possession", {}).get("name", "") == "Contraataque directo":
            focus_points.append({
                "area": "Presión vs Contraataque",
                "advantage": "neutral",
                "description": "Duelo entre la presión alta local y los contraataques directos visitantes"
            })
        elif away_style.get("defensive", {}).get("name", "") == "Presión alta" and \
             home_style.get("possession", {}).get("name", "") == "Contraataque directo":
            focus_points.append({
                "area": "Presión vs Contraataque",
                "advantage": "neutral",
                "description": "Duelo entre la presión alta visitante y los contraataques directos locales"
            })
            
        # Añadir un punto sobre set pieces si ambos equipos son fuertes
        home_set_pieces = home_profile.get("set_pieces", {}).get("name", "")
        away_set_pieces = away_profile.get("set_pieces", {}).get("name", "")
        
        if ("Especialistas" in home_set_pieces or "Corners" in home_set_pieces) and \
           ("Especialistas" in away_set_pieces or "Corners" in away_set_pieces):
            focus_points.append({
                "area": "Balón Parado",
                "advantage": "neutral",
                "description": "Ambos equipos destacan en acciones a balón parado"
            })
        
        return focus_points[:3]  # Limitar a 3 puntos de enfoque
    
    def _parse_formation(self, formation: str) -> List[int]:
        """
        Parsea una formación y devuelve una lista con número de jugadores por línea.
        
        Args:
            formation: String de formación (ej. "4-4-2", "4-3-3")
            
        Returns:
            Lista de enteros con jugadores por línea [defensa, medio, ataque]
        """
        try:
            # Dividir por guiones y convertir a enteros
            return [int(x) for x in formation.split("-")]
        except:
            # Si hay error, devolver una formación por defecto
            return [4, 4, 2]
        
        
    def analyze_tactical_matchup(self, home_profile: Optional[Dict[str, Any]], 
                               away_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza el enfrentamiento táctico entre dos equipos basado en sus perfiles.
        Compatible con equipos de cualquier liga, no solo las principales.
        
        Args:
            home_profile: Perfil táctico del equipo local
            away_profile: Perfil táctico del equipo visitante
            
        Returns:
            Análisis del enfrentamiento táctico con ventajas y desventajas
        """
        # Si falta algún perfil, devolver análisis básico
        if not home_profile or not away_profile:
            return {
                'analysis_quality': 'limited',
                'reason': 'Missing tactical profile data for one or both teams',
                'general_insight': 'Limited tactical analysis available due to insufficient data'
            }
            
        try:
            # Extraer estilos de juego principales
            home_style = {
                'possession': home_profile.get('tactical_style', {}).get('possession', {}).get('name', 'Equilibrado'),
                'defensive': home_profile.get('tactical_style', {}).get('defensive', {}).get('name', 'Bloque medio'),
                'offensive': home_profile.get('tactical_style', {}).get('offensive', {}).get('name', 'Equilibrado'),
                'formation': home_profile.get('formations', [{}])[0].get('formation', '4-4-2'),
                'set_pieces': home_profile.get('set_pieces', {}).get('name', 'Estándar')
            }
            
            away_style = {
                'possession': away_profile.get('tactical_style', {}).get('possession', {}).get('name', 'Equilibrado'),
                'defensive': away_profile.get('tactical_style', {}).get('defensive', {}).get('name', 'Bloque medio'),
                'offensive': away_profile.get('tactical_style', {}).get('offensive', {}).get('name', 'Equilibrado'),
                'formation': away_profile.get('formations', [{}])[0].get('formation', '4-4-2'),
                'set_pieces': away_profile.get('set_pieces', {}).get('name', 'Estándar')
            }
            
            # Analizar ventajas tácticas
            advantages = self._get_tactical_advantages(home_style, away_style)
            
            # Generar narrativa del enfrentamiento
            narrative = self._generate_matchup_narrative(
                home_profile.get('team_name', 'Local'),
                away_profile.get('team_name', 'Visitante'),
                home_style,
                away_style,
                advantages
            )
            
            return {
                'analysis_quality': 'comprehensive',
                'home_advantages': advantages['home_advantages'],
                'away_advantages': advantages['away_advantages'],
                'key_matchups': advantages['key_matchups'],
                'tactical_narrative': narrative,
                'expected_match_flow': self._predict_match_flow(home_style, away_style),
                'data_confidence': 'high' if home_profile.get('data_quality', 'medium') == 'high' 
                                   and away_profile.get('data_quality', 'medium') == 'high' else 'medium'
            }
            
        except Exception as e:
            logger.exception(f"Error en análisis táctico de enfrentamiento: {e}")
            return {
                'analysis_quality': 'error',
                'reason': str(e),
                'general_insight': 'Se produjo un error al analizar el enfrentamiento táctico'
            }
    
    def _get_tactical_advantages(self, home_style: Dict[str, str], away_style: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Identifica ventajas tácticas basadas en los estilos de juego.
        
        Args:
            home_style: Estilo de juego del equipo local
            away_style: Estilo de juego del equipo visitante
            
        Returns:
            Diccionario con ventajas identificadas para cada equipo
        """
        home_advantages = []
        away_advantages = []
        key_matchups = []
        
        # Analizar ventajas de posesión
        if home_style['possession'] == 'Posesión dominante' and away_style['defensive'] == 'Bloque bajo':
            home_advantages.append('Debería dominar la posesión contra un equipo de bloque bajo')
            key_matchups.append('Posesión local vs. Defensa compacta visitante')
        elif away_style['possession'] == 'Posesión dominante' and home_style['defensive'] == 'Bloque bajo':
            away_advantages.append('Debería dominar la posesión contra un equipo de bloque bajo')
            key_matchups.append('Posesión visitante vs. Defensa compacta local')
            
        # Analizar ventajas ofensivas
        if home_style['offensive'] == 'Ofensiva por bandas' and away_style['formation'] in ['3-5-2', '3-4-3']:
            home_advantages.append('Puede explotar espacios en las bandas por la formación rival')
        elif away_style['offensive'] == 'Ofensiva por bandas' and home_style['formation'] in ['3-5-2', '3-4-3']:
            away_advantages.append('Puede explotar espacios en las bandas por la formación rival')
            
        if home_style['offensive'] == 'Contragolpes veloces' and away_style['possession'] == 'Posesión dominante':
            home_advantages.append('Puede aprovechar contragolpes mientras el rival busca posesión')
        elif away_style['offensive'] == 'Contragolpes veloces' and home_style['possession'] == 'Posesión dominante':
            away_advantages.append('Puede aprovechar contragolpes mientras el rival busca posesión')
            
        # Analizar ventajas defensivas
        if home_style['defensive'] == 'Presión alta' and away_style['possession'] == 'Posesión dominante':
            key_matchups.append('Presión alta local vs. Posesión elaborada visitante')
        elif away_style['defensive'] == 'Presión alta' and home_style['possession'] == 'Posesión dominante':
            key_matchups.append('Presión alta visitante vs. Posesión elaborada local')
            
        # Análisis de balón parado
        if home_style['set_pieces'] == 'Especialistas balón parado':
            home_advantages.append('Ventaja en jugadas a balón parado')
        elif away_style['set_pieces'] == 'Especialistas balón parado':
            away_advantages.append('Ventaja en jugadas a balón parado')
            
        # Si hay pocas ventajas identificadas, añadir análisis genérico
        if len(home_advantages) < 1:
            home_advantages.append('Juego equilibrado sin ventajas tácticas claras')
        if len(away_advantages) < 1:
            away_advantages.append('Juego equilibrado sin ventajas tácticas claras')
        if len(key_matchups) < 1:
            key_matchups.append('Choque de estilos equilibrado')
            
        return {
            'home_advantages': home_advantages,
            'away_advantages': away_advantages,
            'key_matchups': key_matchups
        }
    
    def _generate_matchup_narrative(self, home_team: str, away_team: str,
                                  home_style: Dict[str, str], away_style: Dict[str, str],
                                  advantages: Dict[str, List[str]]) -> str:
        """
        Genera una narrativa del enfrentamiento táctico.
        
        Returns:
            Texto narrativo del enfrentamiento
        """
        # Establecer el tono del partido según los estilos
        if home_style['possession'] == 'Posesión dominante' and away_style['defensive'] == 'Bloque bajo':
            tone = f"Esperamos un partido donde {home_team} domine la posesión mientras {away_team} busca contragolpes"
        elif away_style['possession'] == 'Posesión dominante' and home_style['defensive'] == 'Bloque bajo':
            tone = f"Esperamos un partido donde {away_team} domine la posesión mientras {home_team} busca contragolpes"
        elif home_style['defensive'] == 'Presión alta' and away_style['defensive'] == 'Presión alta':
            tone = f"Se espera un partido intenso con ambos equipos presionando alto y disputando cada balón"
        else:
            tone = f"Se prevé un encuentro donde {home_team} con su estilo {home_style['possession']} enfrentará a un {away_team} caracterizado por un juego {away_style['possession']}"
            
        # Construir la narrativa completa
        narrative = f"{tone}. "
        
        # Añadir detalles sobre las ventajas
        if advantages['home_advantages']:
            narrative += f"{home_team} podría tener ventaja porque: {'; '.join(advantages['home_advantages'])}. "
            
        if advantages['away_advantages']:
            narrative += f"{away_team} podría tener ventaja porque: {'; '.join(advantages['away_advantages'])}. "
            
        # Añadir puntos clave del enfrentamiento
        if advantages['key_matchups']:
            narrative += f"Puntos clave del enfrentamiento: {'; '.join(advantages['key_matchups'])}."
            
        return narrative
    
    def _predict_match_flow(self, home_style: Dict[str, str], away_style: Dict[str, str]) -> Dict[str, Any]:
        """
        Predice cómo podría desarrollarse el flujo del partido basado en los estilos.
        
        Returns:
            Predicción del flujo del partido
        """
        # Definir probabilidades de posesión
        possession_map = {
            'Posesión dominante': 0.65,
            'Contraataque directo': 0.4,
            'Presión alta': 0.55,
            'Defensa baja + contraataques': 0.35,
            'Equilibrado': 0.5
        }
        
        # Estimar posesión
        home_possession_base = possession_map.get(home_style['possession'], 0.5)
        away_possession_base = possession_map.get(away_style['possession'], 0.5)
        
        # Normalizar para que sumen 1
        total = home_possession_base + away_possession_base
        home_possession = home_possession_base / total
        away_possession = away_possession_base / total
        
        # Predecir ritmo de juego (bajo, medio, alto)
        if (home_style['possession'] == 'Posesión dominante' and away_style['defensive'] == 'Bloque bajo') or \
           (away_style['possession'] == 'Posesión dominante' and home_style['defensive'] == 'Bloque bajo'):
            pace = 'bajo'
        elif home_style['defensive'] == 'Presión alta' or away_style['defensive'] == 'Presión alta':
            pace = 'alto'
        else:
            pace = 'medio'
            
        # Estimar espacios
        if home_style['defensive'] == 'Bloque bajo' or away_style['defensive'] == 'Bloque bajo':
            spaces = 'reducidos'
        elif home_style['defensive'] == 'Presión alta' and away_style['defensive'] == 'Presión alta':
            spaces = 'amplios'
        else:
            spaces = 'moderados'
            
        return {
            'expected_possession': {'home': round(home_possession * 100), 'away': round(away_possession * 100)},
            'expected_pace': pace,
            'expected_spaces': spaces
        }


# Función de conveniencia para usar directamente
def get_enhanced_tactical_analysis(home_team_id: int, 
                                 away_team_id: int, 
                                 home_team_name: str, 
                                 away_team_name: str,
                                 home_team_stats: Optional[Dict[str, Any]] = None,
                                 away_team_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Genera un análisis táctico mejorado para un partido.
    
    Args:
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        home_team_name: Nombre del equipo local
        away_team_name: Nombre del equipo visitante
        home_team_stats: Estadísticas del equipo local (opcional)
        away_team_stats: Estadísticas del equipo visitante (opcional)
        
    Returns:
        Análisis táctico detallado del partido
    """
    analyzer = EnhancedTacticalAnalyzer()
    return analyzer.generate_match_tactical_analysis(
        home_team_id, away_team_id,
        home_team_name, away_team_name,
        home_team_stats, away_team_stats
    )
