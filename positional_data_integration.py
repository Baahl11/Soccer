"""
Módulo para integración y procesamiento de datos de posicionamiento y tracking en fútbol.

Este módulo implementa la integración de datos de tracking para mejorar las predicciones
de goles, basado en investigaciones del Journal of Sports Analytics (2024) que señalan
que estos datos pueden mejorar las predicciones en un 12-15% al capturar dinámicas tácticas.

Funcionalidades principales:
- Procesamiento de datos de tracking para extraer patrones posicionales
- Cálculo de métricas de espacio controlado y generación de oportunidades
- Medición de eficacia táctica en diferentes zonas del campo
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

# Configuración de matplotlib
import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import joblib
from datetime import datetime
import numpy as np

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes
PITCH_LENGTH = 105  # metros
PITCH_WIDTH = 68    # metros
DEFAULT_MODEL_PATH = "models/positional_metrics_model.pkl"
DATA_CACHE_PATH = "cache/positional_data/"

@dataclass
class PositionalMetrics:
    """Estructura de datos para métricas posicionales de un equipo"""
    team_id: int
    press_intensity: float = 0.0            # Intensidad de presión defensiva
    defensive_compactness: float = 0.0      # Compacidad defensiva
    attacking_width: float = 0.0            # Amplitud ofensiva
    offensive_depth: float = 0.0           # Profundidad ofensiva
    possession_control: float = 0.0        # Control de posesión espacial
    transition_speed: float = 0.0          # Velocidad de transición
    danger_zone_entries: int = 0           # Entradas a zona peligrosa
    effective_playing_area: float = 0.0    # Área de juego efectiva
    defensive_line_height: float = 0.0     # Altura de la línea defensiva
    pass_network_density: float = 0.0      # Densidad de red de pases

class PositionalDataIntegrator:
    """
    Clase principal para la integración y procesamiento de datos posicionales.
    
    Esta clase procesa datos crudos de tracking de jugadores para generar métricas
    significativas que puedan ser utilizadas en modelos predictivos. Proporciona
    interfases para diferentes formatos de proveedores de datos y genera características
    normalizadas para la predicción.
    """
    
    def __init__(self, data_path: str = DATA_CACHE_PATH, model_path: str = DEFAULT_MODEL_PATH):
        """
        Inicializa el integrador de datos posicionales.
        
        Args:
            data_path: Ruta al directorio de caché de datos posicionales
            model_path: Ruta al modelo pre-entrenado para procesar datos posicionales
        """
        self.data_path = data_path
        self.model_path = model_path
        self._ensure_data_directory()
        
        # Cargar modelo pre-entrenado si existe
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Modelo posicional cargado desde {model_path}")
            else:
                self.model = None
                logger.warning(f"No se encontró modelo posicional en {model_path}")
        except Exception as e:
            logger.error(f"Error cargando modelo posicional: {e}")
            self.model = None
        
        # Inicializar escaladores para normalización
        self.feature_scaler = StandardScaler()
        
    def _ensure_data_directory(self) -> None:
        """Asegura que exista el directorio para cache de datos"""
        os.makedirs(self.data_path, exist_ok=True)
        
    def process_raw_tracking_data(self, match_id: int, tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa datos crudos de tracking de un partido.
        
        Args:
            match_id: ID del partido
            tracking_data: Datos crudos de tracking del partido
            
        Returns:
            Métricas procesadas derivadas de los datos de tracking
        """
        logger.info(f"Procesando datos de tracking para partido {match_id}")
        
        try:
            # Extraer datos de los equipos
            home_team_data = tracking_data.get("home_team", {})
            away_team_data = tracking_data.get("away_team", {})
            
            # Verificar datos
            if not home_team_data or not away_team_data:
                logger.warning(f"Datos incompletos para partido {match_id}")
                return {}
                
            # Procesar datos por equipo
            home_metrics = self._calculate_team_positional_metrics(home_team_data, is_home=True)
            away_metrics = self._calculate_team_positional_metrics(away_team_data, is_home=False)
            
            # Calcular métricas comparativas
            comparative_metrics = self._calculate_comparative_metrics(home_metrics, away_metrics)
            
            # Crear resultado final
            result = {
                "match_id": match_id,
                "home_team": {
                    "team_id": home_team_data.get("team_id"),
                    "metrics": self._positional_metrics_to_dict(home_metrics)
                },
                "away_team": {
                    "team_id": away_team_data.get("team_id"),
                    "metrics": self._positional_metrics_to_dict(away_metrics)
                },
                "comparative": comparative_metrics,
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "data_quality": self._assess_data_quality(tracking_data)
                }
            }
            
            # Guardar en caché
            self._cache_processed_data(match_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error procesando datos de tracking para partido {match_id}: {e}")
            return {}
    
    def _calculate_team_positional_metrics(self, team_data: Dict[str, Any], is_home: bool) -> PositionalMetrics:
        """
        Calcula métricas posicionales para un equipo.
        
        Args:
            team_data: Datos de tracking del equipo
            is_home: Indicador si es equipo local
            
        Returns:
            Objeto con métricas posicionales calculadas
        """
        team_id = team_data.get("team_id", 0)
        player_tracks = team_data.get("player_tracks", [])
        
        if not player_tracks:
            return PositionalMetrics(team_id=team_id)
            
        # Inicializar métricas
        metrics = PositionalMetrics(team_id=team_id)
        
        try:
            # Extraer posiciones
            positions = np.array([track.get("mean_position", [0, 0]) for track in player_tracks])
            
            # Calcular compacidad defensiva (desviación estándar de la dispersión espacial)
            metrics.defensive_compactness = float(np.std(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))
            
            # Calcular amplitud ofensiva (distancia máxima entre jugadores en el eje X)
            if positions.size > 0:
                metrics.attacking_width = float(np.max(positions[:, 0]) - np.min(positions[:, 0])) / PITCH_WIDTH
            
            # Calcular profundidad ofensiva (posición media en el eje Y, normalizada)
            attacking_direction = 1 if is_home else -1
            metrics.offensive_depth = float(attacking_direction * np.mean(positions[:, 1])) / PITCH_LENGTH + 0.5
            
            # Altura de la línea defensiva (posición del defensa más retrasado)
            defensive_y = positions[:, 1]
            metrics.defensive_line_height = float(np.min(defensive_y) if is_home else np.max(defensive_y)) / PITCH_LENGTH + 0.5
            
            # Área efectiva de juego (área del polígono convexo)
            if len(positions) >= 3:  # Necesitamos al menos 3 puntos para un polígono
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(positions)
                    metrics.effective_playing_area = float(hull.volume) / (PITCH_LENGTH * PITCH_WIDTH)
                except Exception:
                    metrics.effective_playing_area = 0.0
            
            # Entradas a zona peligrosa (aproximación basada en posiciones en último tercio)
            danger_zone_threshold = PITCH_LENGTH * (0.7 if is_home else 0.3)
            metrics.danger_zone_entries = int(np.sum(positions[:, 1] > danger_zone_threshold if is_home else positions[:, 1] < danger_zone_threshold))
            
            # Aplicar otros cálculos de métricas si hay datos disponibles
            if "pressure_events" in team_data:
                metrics.press_intensity = self._calculate_press_intensity(team_data["pressure_events"])
                
            if "possession_data" in team_data:
                metrics.possession_control = self._calculate_possession_control(team_data["possession_data"], positions)
                
            if "transition_events" in team_data:
                metrics.transition_speed = self._calculate_transition_speed(team_data["transition_events"])
                
            if "pass_network" in team_data:
                metrics.pass_network_density = self._calculate_pass_network_density(team_data["pass_network"])
            
        except Exception as e:
            logger.error(f"Error calculando métricas posicionales para equipo {team_id}: {e}")
        
        return metrics
    
    def _calculate_press_intensity(self, pressure_events: List[Dict[str, Any]]) -> float:
        """Calcula la intensidad de presión basada en eventos de presión"""
        if not pressure_events:
            return 0.0
        
        # La intensidad se calcula como la frecuencia y proximidad de los eventos de presión
        try:
            pressure_count = float(len(pressure_events))
            distances = np.array([float(event.get("distance", 10.0)) for event in pressure_events])
            avg_distance = float(np.maximum(1.0, np.mean(distances))) if len(distances) > 0 else 1.0
            return float(pressure_count / avg_distance * 10.0)  # Normalizar a escala 0-10
        except Exception as e:
            logger.error(f"Error calculando intensidad de presión: {e}")
            return 0.0
    
    def _calculate_possession_control(self, possession_data: Dict[str, Any], positions: np.ndarray) -> float:
        """Calcula el control de posesión espacial basado en datos de posesión y posiciones"""
        if not possession_data:
            return 0.0
        
        try:
            # Control se calcula combinando tiempo de posesión y área dominada
            possession_percentage = possession_data.get("percentage", 0.5)
            field_control = possession_data.get("field_control_percentage", 0.5)
            return float((possession_percentage + field_control) / 2.0)
        except Exception as e:
            logger.error(f"Error calculando control de posesión: {e}")
            return 0.0
    
    def _calculate_transition_speed(self, transition_events: List[Dict[str, Any]]) -> float:
        """Calcula la velocidad de transición basada en eventos de transición"""
        if not transition_events:
            return 0.0
        
        try:
            # La velocidad se mide como el tiempo promedio de transición defensa-ataque
            transition_times = [event.get("duration_seconds", 10.0) for event in transition_events]
            if not transition_times:
                return 0.0
                
            # Invertimos para que valores más altos indiquen transiciones más rápidas
            transition_times_array = np.array([float(t) for t in transition_times])
            avg_time = float(np.maximum(1.0, np.mean(transition_times_array)))
            return float(10.0 / avg_time)  # Normalizar a escala 0-10
        except Exception as e:
            logger.error(f"Error calculando velocidad de transición: {e}")
            return 0.0
    
    def _calculate_pass_network_density(self, pass_network: Dict[str, Any]) -> float:
        """Calcula la densidad de la red de pases"""
        if not pass_network:
            return 0.0
        
        try:
            # La densidad se calcula con el número de pases y conexiones entre jugadores
            connections = pass_network.get("connections", [])
            if not connections:
                return 0.0
                
            num_players = len(pass_network.get("players", []))
            if num_players < 2:
                return 0.0
                
            # Máximo teórico de conexiones para n jugadores: n(n-1)
            max_connections = num_players * (num_players - 1)
            actual_connections = len(connections)
            
            return float(actual_connections / max_connections)
        except Exception as e:
            logger.error(f"Error calculando densidad de red de pases: {e}")
            return 0.0
    
    def _calculate_comparative_metrics(self, home_metrics: PositionalMetrics, away_metrics: PositionalMetrics) -> Dict[str, float]:
        """
        Calcula métricas comparativas entre equipos local y visitante.
        
        Args:
            home_metrics: Métricas del equipo local
            away_metrics: Métricas del equipo visitante
            
        Returns:
            Diccionario de métricas comparativas
        """
        result = {}
        
        # Diferencial de presión
        result["pressure_differential"] = home_metrics.press_intensity - away_metrics.press_intensity
        
        # Diferencial de compacidad
        result["compactness_differential"] = home_metrics.defensive_compactness - away_metrics.defensive_compactness
        
        # Control territorial (combinación de posesión y área efectiva)
        result["territorial_control"] = (home_metrics.possession_control + home_metrics.effective_playing_area) / 2 - \
                                       (away_metrics.possession_control + away_metrics.effective_playing_area) / 2
        
        # Índice de peligro ofensivo
        result["offensive_danger_index"] = (home_metrics.danger_zone_entries / max(1, away_metrics.danger_zone_entries)) - 1
        
        # Índice de transición
        result["transition_advantage"] = home_metrics.transition_speed - away_metrics.transition_speed
        
        return result
    
    def _positional_metrics_to_dict(self, metrics: PositionalMetrics) -> Dict[str, float]:
        """Convierte objeto de métricas posicionales a diccionario"""
        return {
            "press_intensity": round(metrics.press_intensity, 3),
            "defensive_compactness": round(metrics.defensive_compactness, 3),
            "attacking_width": round(metrics.attacking_width, 3),
            "offensive_depth": round(metrics.offensive_depth, 3),
            "possession_control": round(metrics.possession_control, 3),
            "transition_speed": round(metrics.transition_speed, 3),
            "danger_zone_entries": metrics.danger_zone_entries,
            "effective_playing_area": round(metrics.effective_playing_area, 3),
            "defensive_line_height": round(metrics.defensive_line_height, 3),
            "pass_network_density": round(metrics.pass_network_density, 3)
        }
    
    def _assess_data_quality(self, tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa la calidad de los datos de tracking"""
        result = {
            "completeness": 0.0,
            "consistency": 0.0,
            "resolution": 0.0,
            "reliability": 0.0,
            "overall_quality": 0.0
        }
        
        try:
            # Comprobar presencia de equipos
            if "home_team" in tracking_data and "away_team" in tracking_data:
                result["completeness"] = 0.5
                
                # Comprobar datos de jugadores
                home_players = tracking_data["home_team"].get("player_tracks", [])
                away_players = tracking_data["away_team"].get("player_tracks", [])
                
                if home_players and away_players:
                    result["completeness"] = 1.0
                    
            # Verificar consistencia (jugadores esperados por equipo: 11-14)
            home_count = len(tracking_data.get("home_team", {}).get("player_tracks", []))
            away_count = len(tracking_data.get("away_team", {}).get("player_tracks", []))
            
            if 11 <= home_count <= 14 and 11 <= away_count <= 14:
                result["consistency"] = 1.0
            else:
                result["consistency"] = max(0.0, min(1.0, (home_count + away_count) / 22))
            
            # Verificar resolución (frecuencia de muestreo)
            sampling_rate = tracking_data.get("metadata", {}).get("sampling_rate", 0)
            result["resolution"] = min(1.0, sampling_rate / 25.0)  # Normalizado a 25Hz como óptimo
            
            # Verificar fiabilidad (porcentaje de datos válidos)
            valid_percentage = tracking_data.get("metadata", {}).get("valid_percentage", 0)
            result["reliability"] = valid_percentage / 100.0
            
            # Calcular calidad general (promedio ponderado)
            weights = [0.3, 0.25, 0.2, 0.25]  # Pesos para cada factor
            result["overall_quality"] = sum(w * result[key] for w, key in 
                                         zip(weights, ["completeness", "consistency", "resolution", "reliability"]))
            
        except Exception as e:
            logger.error(f"Error evaluando calidad de datos: {e}")
            
        return {k: round(v, 2) for k, v in result.items()}
    
    def _cache_processed_data(self, match_id: int, data: Dict[str, Any]) -> None:
        """Guarda datos procesados en caché"""
        try:
            file_path = os.path.join(self.data_path, f"match_{match_id}_positional.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Datos posicionales guardados para partido {match_id}")
        except Exception as e:
            logger.error(f"Error guardando datos en caché: {e}")
    
    def get_processed_data(self, match_id: int) -> Dict[str, Any]:
        """Recupera datos procesados de caché si existen"""
        try:
            file_path = os.path.join(self.data_path, f"match_{match_id}_positional.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error leyendo datos de caché: {e}")
            return {}
    
    def extract_prediction_features(self, match_id: int, home_team_id: int, away_team_id: int) -> Dict[str, float]:
        """
        Extrae características para predicción a partir de datos posicionales.
        
        Args:
            match_id: ID del partido
            home_team_id: ID del equipo local
            away_team_id: ID del equipo visitante
            
        Returns:
            Diccionario de características para modelos predictivos
        """
        # Intentar cargar datos procesados
        match_data = self.get_processed_data(match_id)
        if not match_data:
            logger.warning(f"No hay datos posicionales disponibles para partido {match_id}")
            return {}
        
        features = {}
        
        try:
            # Extraer métricas por equipo
            home_metrics = match_data.get("home_team", {}).get("metrics", {})
            away_metrics = match_data.get("away_team", {}).get("metrics", {})
            comparative = match_data.get("comparative", {})
            
            # Métricas directas
            if home_metrics and away_metrics:
                # Características de presión
                features["home_press_intensity"] = home_metrics.get("press_intensity", 0)
                features["away_press_intensity"] = away_metrics.get("press_intensity", 0)
                features["press_differential"] = features["home_press_intensity"] - features["away_press_intensity"]
                
                # Características defensivas
                features["home_defensive_compactness"] = home_metrics.get("defensive_compactness", 0)
                features["away_defensive_compactness"] = away_metrics.get("defensive_compactness", 0)
                features["defensive_compactness_ratio"] = (features["home_defensive_compactness"] / 
                                                         max(0.001, features["away_defensive_compactness"]))
                
                # Características ofensivas
                features["home_offensive_depth"] = home_metrics.get("offensive_depth", 0.5)
                features["away_offensive_depth"] = away_metrics.get("offensive_depth", 0.5)
                features["home_attacking_width"] = home_metrics.get("attacking_width", 0)
                features["away_attacking_width"] = away_metrics.get("attacking_width", 0)
                
                # Características de transición
                features["home_transition_speed"] = home_metrics.get("transition_speed", 0)
                features["away_transition_speed"] = away_metrics.get("transition_speed", 0)
                features["transition_speed_advantage"] = comparative.get("transition_advantage", 0)
                
                # Características de dominio territorial
                features["territorial_control"] = comparative.get("territorial_control", 0)
                features["home_effective_area"] = home_metrics.get("effective_playing_area", 0)
                features["away_effective_area"] = away_metrics.get("effective_playing_area", 0)
                
                # Indicadores de peligro
                features["offensive_danger_ratio"] = comparative.get("offensive_danger_index", 0)
                features["home_danger_zone_entries"] = home_metrics.get("danger_zone_entries", 0)
                features["away_danger_zone_entries"] = away_metrics.get("danger_zone_entries", 0)
                
                # Red de pases
                features["home_pass_network_density"] = home_metrics.get("pass_network_density", 0)
                features["away_pass_network_density"] = away_metrics.get("pass_network_density", 0)
                features["pass_network_ratio"] = (features["home_pass_network_density"] / 
                                               max(0.001, features["away_pass_network_density"]))
            
            # Calidad de datos para ponderación
            data_quality = match_data.get("metadata", {}).get("data_quality", {}).get("overall_quality", 0)
            features["positional_data_quality"] = data_quality
            
        except Exception as e:
            logger.error(f"Error extrayendo características posicionales: {e}")
        
        return {k: round(v, 4) for k, v in features.items()}
    def visualize_team_positions(self, match_id: int, half: int = 1) -> Optional[Figure]:
        """
        Genera visualización de posiciones promedio de equipos.
        
        Args:
            match_id: ID del partido
            half: Primera o segunda mitad (1 o 2)
            
        Returns:
            Figura de matplotlib con la visualización o None si hay error
        """
        match_data = self.get_processed_data(match_id)
        if not match_data:
            logger.warning(f"No hay datos posicionales disponibles para partido {match_id}")
            return None
        
        try:
            # Configurar figura
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Dibujar campo
            self._draw_pitch(ax)
            
            # Extraer datos de posiciones
            home_team = match_data.get("home_team", {})
            away_team = match_data.get("away_team", {})
            
            half_key = f"half_{half}"
            
            # Dibujar posiciones del equipo local
            if home_team:
                home_positions = []
                for player in home_team.get("player_tracks", []):
                    if half_key in player:
                        pos = player[half_key].get("mean_position", [0, 0])
                        jersey = player.get("jersey_number", "")
                        home_positions.append(pos)
                        ax.plot(pos[0], pos[1], 'bo', markersize=8)
                        ax.text(pos[0] + 1, pos[1] + 1, str(jersey), color='blue', fontsize=8)
                
                # Dibujar polígono convexo del equipo local
                if len(home_positions) >= 3:
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(np.array(home_positions))
                        for simplex in hull.simplices:
                            plt.plot(np.array(home_positions)[simplex, 0], 
                                     np.array(home_positions)[simplex, 1], 'b-', alpha=0.4)
                    except Exception:
                        pass
            
            # Dibujar posiciones del equipo visitante
            if away_team:
                away_positions = []
                for player in away_team.get("player_tracks", []):
                    if half_key in player:
                        pos = player[half_key].get("mean_position", [0, 0])
                        jersey = player.get("jersey_number", "")
                        away_positions.append(pos)
                        ax.plot(pos[0], pos[1], 'ro', markersize=8)
                        ax.text(pos[0] + 1, pos[1] + 1, str(jersey), color='red', fontsize=8)
                
                # Dibujar polígono convexo del equipo visitante
                if len(away_positions) >= 3:
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(np.array(away_positions))
                        for simplex in hull.simplices:
                            plt.plot(np.array(away_positions)[simplex, 0], 
                                     np.array(away_positions)[simplex, 1], 'r-', alpha=0.4)
                    except Exception:
                        pass
            
            # Añadir leyenda y título
            home_team_name = home_team.get("team_name", "Local")
            away_team_name = away_team.get("team_name", "Visitante")
            plt.title(f"Posiciones medias: {home_team_name} vs {away_team_name} - {half}ª mitad")
            plt.legend([f"{home_team_name} (Local)", f"{away_team_name} (Visitante)"], 
                      loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generando visualización: {e}")
            return None
    
    def _draw_pitch(self, ax: Axes) -> None:
        """Dibuja un campo de fútbol en los ejes proporcionados"""
        # Límites del campo
        ax.set_xlim(0, PITCH_WIDTH)
        ax.set_ylim(0, PITCH_LENGTH)
        
        # Rectángulo principal
        rect = Rectangle((0, 0), PITCH_WIDTH, PITCH_LENGTH, fill=False, color='green')
        ax.add_patch(rect)
        
        # Línea central
        plt.plot([0, PITCH_WIDTH], [PITCH_LENGTH/2, PITCH_LENGTH/2], 'green')
        
        # Círculo central
        circle = Circle((PITCH_WIDTH/2, PITCH_LENGTH/2), 9.15, fill=False, color='green')
        ax.add_patch(circle)
        
        # Áreas de penalti
        rect_penalty_1 = Rectangle((PITCH_WIDTH/2 - 20.16, 0), 
                                     40.32, 16.5, fill=False, color='green')
        rect_penalty_2 = Rectangle((PITCH_WIDTH/2 - 20.16, PITCH_LENGTH - 16.5), 
                                     40.32, 16.5, fill=False, color='green')
        ax.add_patch(rect_penalty_1)
        ax.add_patch(rect_penalty_2)
        
        # Áreas de portería
        rect_goal_1 = Rectangle((PITCH_WIDTH/2 - 9.16, 0), 
                                   18.32, 5.5, fill=False, color='green')
        rect_goal_2 = Rectangle((PITCH_WIDTH/2 - 9.16, PITCH_LENGTH - 5.5), 
                                   18.32, 5.5, fill=False, color='green')
        ax.add_patch(rect_goal_1)
        ax.add_patch(rect_goal_2)
        
        # Punto de penalti
        plt.plot(PITCH_WIDTH/2, 11, 'go', markersize=2)
        plt.plot(PITCH_WIDTH/2, PITCH_LENGTH - 11, 'go', markersize=2)
        
        # Eliminar ejes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Color de fondo
        ax.set_facecolor('#a8e495')  # Verde claro

    def analyze_match_positioning(self, match_id: int) -> Dict[str, Any]:
        """
        Realiza un análisis completo de posicionamiento para un partido.
        
        Args:
            match_id: ID del partido
            
        Returns:
            Diccionario con análisis y conclusiones tácticas
        """
        match_data = self.get_processed_data(match_id)
        if not match_data:
            logger.warning(f"No hay datos posicionales disponibles para partido {match_id}")
            return {}
            
        home_team = match_data.get("home_team", {})
        away_team = match_data.get("away_team", {})
        home_metrics = home_team.get("metrics", {})
        away_metrics = away_team.get("metrics", {})
        
        if not home_metrics or not away_metrics:
            return {}
            
        analysis = {
            "match_id": match_id,
            "home_team_id": home_team.get("team_id"),
            "away_team_id": away_team.get("team_id"),
            "tactical_insights": [],
            "key_metrics": {},
            "prediction_impact": {}
        }
        
        # Analizar información táctica
        insights = []
        
        # Analizar presión
        if home_metrics.get("press_intensity", 0) > away_metrics.get("press_intensity", 0) * 1.2:
            insights.append({
                "type": "pressure",
                "description": "El equipo local ejerce una presión significativamente mayor",
                "advantage": "home",
                "strength": "high" if home_metrics.get("press_intensity", 0) > 7 else "medium"
            })
        elif away_metrics.get("press_intensity", 0) > home_metrics.get("press_intensity", 0) * 1.2:
            insights.append({
                "type": "pressure",
                "description": "El equipo visitante ejerce una presión significativamente mayor",
                "advantage": "away",
                "strength": "high" if away_metrics.get("press_intensity", 0) > 7 else "medium"
            })
            
        # Analizar compacidad defensiva
        home_compactness = home_metrics.get("defensive_compactness", 0)
        away_compactness = away_metrics.get("defensive_compactness", 0)
        compact_ratio = home_compactness / max(0.001, away_compactness)
        
        if compact_ratio < 0.8:
            insights.append({
                "type": "defensive_organization",
                "description": "El equipo local mantiene una estructura defensiva más compacta",
                "advantage": "home",
                "strength": "high" if compact_ratio < 0.6 else "medium"
            })
        elif compact_ratio > 1.2:
            insights.append({
                "type": "defensive_organization",
                "description": "El equipo visitante mantiene una estructura defensiva más compacta",
                "advantage": "away",
                "strength": "high" if compact_ratio > 1.4 else "medium"
            })
            
        # Analizar control territorial
        territorial = match_data.get("comparative", {}).get("territorial_control", 0)
        if territorial > 0.15:
            insights.append({
                "type": "territorial_control",
                "description": "El equipo local domina territorial y espacialmente",
                "advantage": "home",
                "strength": "high" if territorial > 0.3 else "medium"
            })
        elif territorial < -0.15:
            insights.append({
                "type": "territorial_control",
                "description": "El equipo visitante domina territorial y espacialmente",
                "advantage": "away",
                "strength": "high" if territorial < -0.3 else "medium"
            })
            
        # Analizar peligrosidad ofensiva
        danger_index = match_data.get("comparative", {}).get("offensive_danger_index", 0)
        if danger_index > 0.2:
            insights.append({
                "type": "offensive_danger",
                "description": "El equipo local genera significativamente más situaciones de peligro",
                "advantage": "home",
                "strength": "high" if danger_index > 0.5 else "medium"
            })
        elif danger_index < -0.2:
            insights.append({
                "type": "offensive_danger",
                "description": "El equipo visitante genera significativamente más situaciones de peligro",
                "advantage": "away",
                "strength": "high" if danger_index < -0.5 else "medium"
            })
            
        # Analizar velocidad de transición
        transition_adv = match_data.get("comparative", {}).get("transition_advantage", 0)
        if transition_adv > 1:
            insights.append({
                "type": "transitions",
                "description": "El equipo local realiza transiciones ataque-defensa más rápidas",
                "advantage": "home",
                "strength": "high" if transition_adv > 2 else "medium"
            })
        elif transition_adv < -1:
            insights.append({
                "type": "transitions",
                "description": "El equipo visitante realiza transiciones ataque-defensa más rápidas",
                "advantage": "away",
                "strength": "high" if transition_adv < -2 else "medium"
            })
            
        # Añadir insights al análisis
        analysis["tactical_insights"] = insights
        
        # Métricas clave para predicción
        key_metrics = {}
        for metric in ["press_intensity", "defensive_compactness", "effective_playing_area", "transition_speed"]:
            key_metrics[f"home_{metric}"] = home_metrics.get(metric, 0)
            key_metrics[f"away_{metric}"] = away_metrics.get(metric, 0)
            key_metrics[f"{metric}_ratio"] = (home_metrics.get(metric, 0) / 
                                            max(0.001, away_metrics.get(metric, 0)))
            
        for metric in ["territorial_control", "offensive_danger_index", "transition_advantage"]:
            key_metrics[metric] = match_data.get("comparative", {}).get(metric, 0)
            
        analysis["key_metrics"] = {k: round(v, 3) for k, v in key_metrics.items()}
        
        # Estimar impacto en predicciones
        home_advantage = 0
        away_advantage = 0
        
        for insight in insights:
            if insight["advantage"] == "home":
                home_advantage += 1.0 if insight["strength"] == "high" else 0.5
            else:
                away_advantage += 1.0 if insight["strength"] == "high" else 0.5
                
        # Calcular impacto neto
        net_impact = home_advantage - away_advantage
        
        analysis["prediction_impact"] = {
            "home_goals_modifier": round(0.3 * net_impact, 2),
            "away_goals_modifier": round(-0.25 * net_impact, 2),
            "confidence": min(1.0, match_data.get("metadata", {}).get("data_quality", {}).get("overall_quality", 0))
        }

        return analysis
    
    def integrate_with_prediction(self, match_id: int, base_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integra análisis posicional con predicción base.
        
        Args:
            match_id: ID del partido
            base_prediction: Predicción base a modificar
            
        Returns:
            Predicción modificada con datos posicionales
        """
        if not base_prediction or "goals" not in base_prediction:
            logger.warning("Predicción base inválida para integración posicional")
            return base_prediction
            
        # Obtener análisis posicional
        analysis = self.analyze_match_positioning(match_id)
        if not analysis or "prediction_impact" not in analysis:
            logger.info(f"No hay análisis posicional disponible para partido {match_id}")
            return base_prediction
            
        # Crear copia para modificar
        modified = base_prediction.copy()
        
        # Extraer impacto
        impact = analysis.get("prediction_impact", {})
        try:
            home_mod = float(impact.get("home_goals_modifier", 0))
            away_mod = float(impact.get("away_goals_modifier", 0))
            confidence = float(impact.get("confidence", 0.5))
            
            # Aplicar modificadores a predicción base
            if "predicted_home_goals" in modified.get("goals", {}):
                current = float(modified["goals"]["predicted_home_goals"])
                modified["goals"]["predicted_home_goals"] = max(0, current + home_mod)
                
            if "predicted_away_goals" in modified.get("goals", {}):
                current = float(modified["goals"]["predicted_away_goals"])
                modified["goals"]["predicted_away_goals"] = max(0, current + away_mod)
            
            # Agregar metadatos del análisis
            if "metadata" not in modified:
                modified["metadata"] = {}
                
            modified["metadata"]["positional_analysis"] = {
                "confidence": confidence,
                "modifiers": {
                    "home": home_mod,
                    "away": away_mod
                }
            }
            
            # Incluir insights tácticos
            modified["tactical_insights"] = analysis.get("tactical_insights", [])
            
        except Exception as e:
            logger.error(f"Error integrando datos posicionales: {e}")
            return base_prediction
            
        return modified

    def train_positional_model(self, training_data: List[Dict[str, Any]], target: str = 'goals') -> bool:
        """
        Entrena un modelo de predicción basado en datos posicionales.
        
        Args:
            training_data: Lista de diccionarios con datos de entrenamiento
            target: Objetivo de predicción ('goals', 'corners', etc.)
            
        Returns:
            True si el entrenamiento fue exitoso, False en caso contrario
        """
        if not training_data:
            logger.error("No hay datos para entrenar el modelo posicional")
            return False
            
        try:
            # Extraer datos y etiquetas
            features = []
            labels = []
            
            for match in training_data:
                # Extraer características posicionales
                match_features = {}
                
                if "positional_metrics" in match:
                    metrics = match["positional_metrics"]
                    
                    # Extraer métricas relevantes
                    for team_type in ["home", "away"]:
                        if team_type in metrics:
                            for key, value in metrics[team_type].items():
                                match_features[f"{team_type}_{key}"] = value
                    
                    # Extraer métricas comparativas
                    if "comparative" in metrics:
                        for key, value in metrics["comparative"].items():
                            match_features[key] = value
                            
                    # Añadir a conjunto de entrenamiento si tenemos suficientes métricas
                    if len(match_features) >= 5:
                        features.append(match_features)
                        
                        # Extraer etiqueta según objetivo
                        if target == 'goals':
                            labels.append(match.get("result", {}).get("home_goals", 0))
                        elif target == 'corners':
                            labels.append(match.get("stats", {}).get("corners", {}).get("home", 0))
                        else:
                            labels.append(match.get(target, 0))
            
            if len(features) < 10:
                logger.warning(f"Datos insuficientes para entrenar modelo posicional: {len(features)} muestras")
                return False
                
            # Preparar datos para entrenamiento
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            
            # Convertir a DataFrame
            X = pd.DataFrame(features)
            y = np.array(labels)
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Normalizar características
            self.feature_scaler = StandardScaler()
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Entrenar modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluar modelo
            test_score = model.score(X_test_scaled, y_test)
            logger.info(f"Modelo posicional entrenado con R² de {test_score:.3f}")
            
            # Guardar modelo
            self.model = model
            joblib.dump((model, self.feature_scaler), self.model_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error entrenando modelo posicional: {e}")
            return False

def create_sample_tracking_data(match_id: int, home_team_id: int, away_team_id: int) -> Dict[str, Any]:
    """
    Crea datos de muestra para pruebas.
    
    Args:
        match_id: ID del partido
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        
    Returns:
        Datos de tracking sintéticos para prueba
    """
    np.random.seed(match_id)
    
    # Crear datos de jugadores
    def create_player_tracks(team_id, is_home):
        player_tracks = []
        for i in range(11):
            jersey = i + 1
            
            # Posiciones base según si es local o visitante
            y_base = PITCH_LENGTH * (0.35 if is_home else 0.65)
            
            # Añadir variación según posición
            if i == 0:  # Portero
                pos_x = PITCH_WIDTH / 2
                pos_y = 5 if is_home else PITCH_LENGTH - 5
            elif i <= 4:  # Defensas
                pos_x = 10 + i * 10
                pos_y = y_base - 15 * (-1 if is_home else 1)
            elif i <= 8:  # Centrocampistas
                pos_x = 5 + (i - 4) * 15
                pos_y = y_base
            else:  # Delanteros
                pos_x = 15 + (i - 8) * 20
                pos_y = y_base + 15 * (-1 if is_home else 1)
                
            # Añadir ruido
            pos_x += np.random.normal(0, 3)
            pos_y += np.random.normal(0, 3)
            
            # Confinar al campo
            pos_x = max(0, min(PITCH_WIDTH, pos_x))
            pos_y = max(0, min(PITCH_LENGTH, pos_y))
            
            player_tracks.append({
                "player_id": team_id * 100 + jersey,
                "jersey_number": jersey,
                "mean_position": [pos_x, pos_y],
                "half_1": {
                    "mean_position": [pos_x, pos_y],
                    "distance_covered": np.random.uniform(4000, 6000)
                },
                "half_2": {
                    "mean_position": [pos_x + np.random.normal(0, 2), 
                                    pos_y + np.random.normal(0, 2)],
                    "distance_covered": np.random.uniform(3800, 5800)
                }
            })
        return player_tracks
    
    # Crear eventos de presión
    def create_pressure_events(high_press):
        count = np.random.randint(15, 35) if high_press else np.random.randint(5, 15)
        events = []
        for _ in range(count):
            events.append({
                "time": np.random.randint(1, 90),
                "position": [np.random.uniform(0, PITCH_WIDTH), 
                           np.random.uniform(0, PITCH_LENGTH)],
                "distance": np.random.uniform(2, 10),
                "duration": np.random.uniform(1, 3)
            })
        return events
    
    # Crear eventos de transición
    def create_transition_events(fast_transitions):
        count = np.random.randint(10, 20)
        events = []
        for _ in range(count):
            base_duration = np.random.uniform(4, 15)
            if fast_transitions:
                duration = base_duration * 0.7
            else:
                duration = base_duration
                
            events.append({
                "time": np.random.randint(1, 90),
                "starting_position": [np.random.uniform(0, PITCH_WIDTH), 
                                    np.random.uniform(0, PITCH_LENGTH)],
                "ending_position": [np.random.uniform(0, PITCH_WIDTH), 
                                  np.random.uniform(0, PITCH_LENGTH)],
                "duration_seconds": duration,
                "players_involved": np.random.randint(2, 6)
            })
        return events
    
    # Determinar estilos de juego para el partido
    home_high_press = np.random.random() > 0.4  # 60% probabilidad de presión alta
    home_fast_transitions = np.random.random() > 0.5
    away_high_press = np.random.random() > 0.6  # 40% probabilidad de presión alta para visitante
    away_fast_transitions = np.random.random() > 0.5
    
    # Crear datos de tracking
    tracking_data = {
        "match_id": match_id,
        "home_team": {
            "team_id": home_team_id,
            "team_name": f"Team {home_team_id}",
            "player_tracks": create_player_tracks(home_team_id, True),
            "pressure_events": create_pressure_events(home_high_press),
            "transition_events": create_transition_events(home_fast_transitions),
            "possession_data": {
                "percentage": np.random.uniform(0.35, 0.65),
                "field_control_percentage": np.random.uniform(0.35, 0.65)
            }
        },
        "away_team": {
            "team_id": away_team_id,
            "team_name": f"Team {away_team_id}",
            "player_tracks": create_player_tracks(away_team_id, False),
            "pressure_events": create_pressure_events(away_high_press),
            "transition_events": create_transition_events(away_fast_transitions),
            "possession_data": {
                "percentage": np.random.uniform(0.35, 0.65),
                "field_control_percentage": np.random.uniform(0.35, 0.65)
            }
        },
        "metadata": {
            "sampling_rate": 25,
            "valid_percentage": np.random.uniform(85, 98),
            "tracking_system": "OptaSports"
        }
    }
    
    return tracking_data
    
def main():
    """Función principal para demostración"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Crear integrador
    integrator = PositionalDataIntegrator()
    
    # Generar datos de ejemplo
    match_id = 12345
    home_team_id = 55
    away_team_id = 65
    
    print(f"Generando datos de tracking para partido {match_id}")
    tracking_data = create_sample_tracking_data(match_id, home_team_id, away_team_id)
    
    # Procesar datos
    print("Procesando datos de tracking...")
    processed = integrator.process_raw_tracking_data(match_id, tracking_data)
    
    # Mostrar métricas procesadas
    print("\nMétricas posicionales procesadas:")
    home_metrics = processed.get("home_team", {}).get("metrics", {})
    away_metrics = processed.get("away_team", {}).get("metrics", {})
    
    print("Equipo local:")
    for k, v in home_metrics.items():
        print(f"  - {k}: {v}")
        
    print("\nEquipo visitante:")
    for k, v in away_metrics.items():
        print(f"  - {k}: {v}")
        
    # Analizar partido
    print("\nAnalizando patrón de juego...")
    analysis = integrator.analyze_match_positioning(match_id)
    
    print("\nInsights tácticos:")
    for insight in analysis.get("tactical_insights", []):
        advantage = "local" if insight["advantage"] == "home" else "visitante"
        print(f"  - {insight['description']} (ventaja {advantage}, intensidad {insight['strength']})")
    
    # Mostrar impacto en predicción
    impact = analysis.get("prediction_impact", {})
    print(f"\nImpacto en predicción: Home +{impact.get('home_goals_modifier', 0)}, Away {impact.get('away_goals_modifier', 0)}")
    
    # Crear visualización
    print("\nGenerando visualización de posiciones...")
    fig = integrator.visualize_team_positions(match_id)
    if fig:
        plt.tight_layout()
        plt.savefig(f"match_{match_id}_positions.png")
        print(f"Visualización guardada como match_{match_id}_positions.png")
    
if __name__ == "__main__":
    main()
