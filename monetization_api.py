#!/usr/bin/env python3
"""
Monetization API - Soccer Predictions Subscription Platform

Este módulo implementa la API principal para la plataforma de suscripción
que muestra predicciones de fútbol de las próximas 24 horas.
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sqlite3
import os
from dataclasses import dataclass, asdict
from enum import Enum

# Import our existing prediction systems
from enhanced_match_winner import predict_with_enhanced_system
from prediction_integration import make_integrated_prediction
from voting_ensemble_corners import predict_corners_optimized
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

class SubscriptionTier(Enum):
    BASIC = "basic"
    PRO = "pro"
    PREMIUM = "premium"
    VIP = "vip"

@dataclass
class MatchPrediction:
    """Estructura de datos para predicciones de partidos."""
    fixture_id: int
    home_team: str
    away_team: str
    league: str
    match_time: str
    predictions_1x2: Dict[str, float]
    corners_prediction: float
    goals_prediction: float
    confidence: float
    value_bets: List[Dict[str, Any]]
    recommendation: str

class PredictionService:
    """Servicio principal para generar predicciones."""
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
    
    def get_todays_fixtures(self) -> List[Dict[str, Any]]:
        """Obtiene todos los partidos de las próximas 24 horas."""
        # Mock data para desarrollo - en producción esto vendría de la API de fútbol
        mock_fixtures = [
            {
                "fixture_id": 1001,
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "home_team_id": 541,
                "away_team_id": 529,
                "league": "La Liga",
                "league_id": 140,
                "match_time": "2024-12-01 21:00:00",
                "venue": "Santiago Bernabéu"
            },
            {
                "fixture_id": 1002,
                "home_team": "Manchester United",
                "away_team": "Liverpool",
                "home_team_id": 33,
                "away_team_id": 40,
                "league": "Premier League",
                "league_id": 39,
                "match_time": "2024-12-01 17:30:00",
                "venue": "Old Trafford"
            },
            {
                "fixture_id": 1003,
                "home_team": "Juventus",
                "away_team": "AC Milan",
                "home_team_id": 496,
                "away_team_id": 489,
                "league": "Serie A",
                "league_id": 135,
                "match_time": "2024-12-01 20:45:00",
                "venue": "Allianz Stadium"
            },
            {
                "fixture_id": 1004,
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "home_team_id": 157,
                "away_team_id": 165,
                "league": "Bundesliga",
                "league_id": 78,
                "match_time": "2024-12-01 18:30:00",
                "venue": "Allianz Arena"
            },
            {
                "fixture_id": 1005,
                "home_team": "PSG",
                "away_team": "Marseille",
                "home_team_id": 85,
                "away_team_id": 81,
                "league": "Ligue 1",
                "league_id": 61,
                "match_time": "2024-12-01 20:00:00",
                "venue": "Parc des Princes"
            }
        ]
        
        return mock_fixtures
    
    def generate_prediction_for_match(self, fixture: Dict[str, Any]) -> MatchPrediction:
        """Genera predicción completa para un partido."""
        try:
            # Usar nuestro sistema de predicción existente
            enhanced_prediction = predict_with_enhanced_system(
                home_team_id=fixture["home_team_id"],
                away_team_id=fixture["away_team_id"],
                league_id=fixture.get("league_id")
            )
            
            # Predicción de corners
            try:
                corners_result = predict_corners_optimized(
                    home_team_id=fixture["home_team_id"],
                    away_team_id=fixture["away_team_id"],
                    league_id=fixture.get("league_id", 39)
                )
                corners_prediction = corners_result.get('ensemble_prediction', 9.5)
            except Exception as e:
                logger.warning(f"Error en predicción de corners: {e}")
                corners_prediction = 9.5
            
            # Extraer probabilidades 1X2
            probabilities = enhanced_prediction.get('probabilities', {})
            home_win_prob = probabilities.get('home_win', 0.33) * 100
            draw_prob = probabilities.get('draw', 0.33) * 100
            away_win_prob = probabilities.get('away_win', 0.33) * 100
            
            # Calcular confianza general
            max_prob = max(home_win_prob, draw_prob, away_win_prob)
            confidence = min(max_prob, 95)  # Cap at 95%
            
            # Simular predicción de goals (en producción usaríamos el sistema real)
            goals_prediction = 2.3 + (max_prob - 33) * 0.02
            
            # Detectar value bets
            value_bets = self._detect_value_bets(probabilities, fixture)
            
            # Generar recomendación
            recommendation = self._generate_recommendation(probabilities, confidence, value_bets)
            
            return MatchPrediction(
                fixture_id=fixture["fixture_id"],
                home_team=fixture["home_team"],
                away_team=fixture["away_team"],
                league=fixture["league"],
                match_time=fixture["match_time"],
                predictions_1x2={
                    "home_win": round(home_win_prob, 1),
                    "draw": round(draw_prob, 1),
                    "away_win": round(away_win_prob, 1)
                },
                corners_prediction=round(corners_prediction, 1),
                goals_prediction=round(goals_prediction, 1),
                confidence=round(confidence, 1),
                value_bets=value_bets,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error generando predicción para partido {fixture['fixture_id']}: {e}")
            # Fallback prediction
            return MatchPrediction(
                fixture_id=fixture["fixture_id"],
                home_team=fixture["home_team"],
                away_team=fixture["away_team"],
                league=fixture["league"],
                match_time=fixture["match_time"],
                predictions_1x2={"home_win": 33.3, "draw": 33.3, "away_win": 33.3},
                corners_prediction=9.5,
                goals_prediction=2.5,
                confidence=50.0,
                value_bets=[],
                recommendation="Análisis en progreso"
            )
    
    def _detect_value_bets(self, probabilities: Dict[str, float], fixture: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detecta value bets comparando nuestras probabilidades con odds simuladas."""
        value_bets = []
        
        # Simular odds de casas de apuestas (en producción vendría de APIs reales)
        mock_odds = {
            'home_win': 2.10,
            'draw': 3.40,
            'away_win': 3.20
        }
        
        for outcome, our_prob in probabilities.items():
            if our_prob > 0:
                odds = mock_odds.get(outcome, 2.0)
                implied_prob = 1 / odds
                
                # Value bet si nuestra probabilidad es 10% mayor que la implícita
                if our_prob > implied_prob * 1.1:
                    value_percentage = ((our_prob / implied_prob) - 1) * 100
                    value_bets.append({
                        'market': outcome,
                        'our_probability': round(our_prob * 100, 1),
                        'implied_probability': round(implied_prob * 100, 1),
                        'value_percentage': round(value_percentage, 1),
                        'odds': odds,
                        'recommendation': 'STRONG' if value_percentage > 20 else 'MODERATE'
                    })
        
        return value_bets
    
    def _generate_recommendation(self, probabilities: Dict[str, float], confidence: float, value_bets: List[Dict[str, Any]]) -> str:
        """Genera recomendación basada en predicciones y value bets."""
        if confidence > 80:
            strongest_outcome = max(probabilities.items(), key=lambda x: x[1])
            return f"Alta confianza en {strongest_outcome[0]} ({confidence:.1f}%)"
        elif value_bets:
            best_value = max(value_bets, key=lambda x: x['value_percentage'])
            return f"Value bet detectado: {best_value['market']} (+{best_value['value_percentage']:.1f}%)"
        elif confidence > 60:
            return f"Predicción moderada (confianza {confidence:.1f}%)"
        else:
            return "Partido impredecible - precaución recomendada"

# Initialize prediction service
prediction_service = PredictionService()

# API Endpoints
@app.route('/api/matches/today')
def get_matches_today():
    """Obtiene todos los partidos de hoy con predicciones."""
    try:
        # Get today's fixtures
        fixtures = prediction_service.get_todays_fixtures()
        
        # Generate predictions for each match
        predictions = []
        for fixture in fixtures:
            prediction = prediction_service.generate_prediction_for_match(fixture)
            predictions.append(asdict(prediction))
        
        # Sort by match time
        predictions.sort(key=lambda x: x['match_time'])
        
        return jsonify({
            'success': True,
            'data': predictions,
            'total_matches': len(predictions),
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error en /api/matches/today: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/matches/<int:fixture_id>')
def get_match_details(fixture_id: int):
    """Obtiene detalles completos de un partido específico."""
    try:
        fixtures = prediction_service.get_todays_fixtures()
        fixture = next((f for f in fixtures if f['fixture_id'] == fixture_id), None)
        
        if not fixture:
            return jsonify({'success': False, 'error': 'Partido no encontrado'}), 404
        
        prediction = prediction_service.generate_prediction_for_match(fixture)
        
        # Add extra analysis for detailed view
        detailed_prediction = asdict(prediction)
        detailed_prediction['detailed_analysis'] = {
            'form_analysis': f"Análisis de forma para {fixture['home_team']} vs {fixture['away_team']}",
            'head_to_head': "Últimos 5 enfrentamientos: 2-1-2",
            'key_factors': [
                "Partido de alta intensidad esperado",
                "Ambos equipos en buena forma",
                "Factor campo importante"
            ]
        }
        
        return jsonify({
            'success': True,
            'data': detailed_prediction
        })
        
    except Exception as e:
        logger.error(f"Error en /api/matches/{fixture_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/leagues/active')
def get_active_leagues():
    """Obtiene ligas activas con partidos hoy."""
    try:
        fixtures = prediction_service.get_todays_fixtures()
        leagues = {}
        
        for fixture in fixtures:
            league = fixture['league']
            if league not in leagues:
                leagues[league] = {
                    'name': league,
                    'matches_count': 0,
                    'matches': []
                }
            leagues[league]['matches_count'] += 1
            leagues[league]['matches'].append({
                'fixture_id': fixture['fixture_id'],
                'home_team': fixture['home_team'],
                'away_team': fixture['away_team'],
                'match_time': fixture['match_time']
            })
        
        return jsonify({
            'success': True,
            'data': list(leagues.values())
        })
        
    except Exception as e:
        logger.error(f"Error en /api/leagues/active: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/value-bets')
def get_value_bets():
    """Obtiene todos los value bets del día."""
    try:
        fixtures = prediction_service.get_todays_fixtures()
        all_value_bets = []
        
        for fixture in fixtures:
            prediction = prediction_service.generate_prediction_for_match(fixture)
            for value_bet in prediction.value_bets:
                value_bet['fixture_id'] = fixture['fixture_id']
                value_bet['match'] = f"{fixture['home_team']} vs {fixture['away_team']}"
                value_bet['league'] = fixture['league']
                value_bet['match_time'] = fixture['match_time']
                all_value_bets.append(value_bet)
        
        # Sort by value percentage
        all_value_bets.sort(key=lambda x: x['value_percentage'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': all_value_bets,
            'total_value_bets': len(all_value_bets)
        })
        
    except Exception as e:
        logger.error(f"Error en /api/value-bets: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/high-confidence')
def get_high_confidence_predictions():
    """Obtiene predicciones de alta confianza (>75%)."""
    try:
        fixtures = prediction_service.get_todays_fixtures()
        high_confidence = []
        
        for fixture in fixtures:
            prediction = prediction_service.generate_prediction_for_match(fixture)
            if prediction.confidence > 75:
                high_confidence.append(asdict(prediction))
        
        # Sort by confidence
        high_confidence.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': high_confidence,
            'total_matches': len(high_confidence)
        })
        
    except Exception as e:
        logger.error(f"Error en /api/high-confidence: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Web Dashboard Route
@app.route('/')
def dashboard():
    """Renderiza el dashboard principal."""
    dashboard_html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soccer Predictions - Plataforma de Suscripción</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-shadow { box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .confidence-high { border-left: 4px solid #10b981; }
        .confidence-medium { border-left: 4px solid #f59e0b; }
        .confidence-low { border-left: 4px solid #ef4444; }
    </style>
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <header class="gradient-bg text-white py-6">
        <div class="container mx-auto px-4">
            <div class="flex justify-between items-center">
                <h1 class="text-3xl font-bold">⚽ Soccer Predictions</h1>
                <div class="flex items-center space-x-4">
                    <span class="bg-white/20 px-3 py-1 rounded-full text-sm">Demo Mode</span>
                    <button class="bg-white text-purple-600 px-4 py-2 rounded-lg font-semibold hover:bg-gray-100">
                        Suscribirse
                    </button>
                </div>
            </div>
        </div>
    </header>

    <!-- Dashboard Content -->
    <div class="container mx-auto px-4 py-8">
        <!-- Stats Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg p-6 card-shadow">
                <div class="flex items-center">
                    <div class="p-3 bg-blue-100 rounded-full">
                        <i data-lucide="calendar" class="w-6 h-6 text-blue-600"></i>
                    </div>
                    <div class="ml-4">
                        <p class="text-gray-600">Partidos Hoy</p>
                        <p class="text-2xl font-bold" id="total-matches">-</p>
                    </div>
                </div>
            </div>
            <div class="bg-white rounded-lg p-6 card-shadow">
                <div class="flex items-center">
                    <div class="p-3 bg-green-100 rounded-full">
                        <i data-lucide="trending-up" class="w-6 h-6 text-green-600"></i>
                    </div>
                    <div class="ml-4">
                        <p class="text-gray-600">Alta Confianza</p>
                        <p class="text-2xl font-bold" id="high-confidence">-</p>
                    </div>
                </div>
            </div>
            <div class="bg-white rounded-lg p-6 card-shadow">
                <div class="flex items-center">
                    <div class="p-3 bg-yellow-100 rounded-full">
                        <i data-lucide="target" class="w-6 h-6 text-yellow-600"></i>
                    </div>
                    <div class="ml-4">
                        <p class="text-gray-600">Value Bets</p>
                        <p class="text-2xl font-bold" id="value-bets">-</p>
                    </div>
                </div>
            </div>
            <div class="bg-white rounded-lg p-6 card-shadow">
                <div class="flex items-center">
                    <div class="p-3 bg-purple-100 rounded-full">
                        <i data-lucide="clock" class="w-6 h-6 text-purple-600"></i>
                    </div>
                    <div class="ml-4">
                        <p class="text-gray-600">Próximo Partido</p>
                        <p class="text-sm font-bold" id="next-match">-</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Filters -->
        <div class="bg-white rounded-lg p-6 card-shadow mb-8">
            <h3 class="text-lg font-semibold mb-4">Filtros</h3>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <select id="league-filter" class="border rounded-lg px-3 py-2">
                    <option value="all">Todas las Ligas</option>
                </select>
                <select id="confidence-filter" class="border rounded-lg px-3 py-2">
                    <option value="all">Toda Confianza</option>
                    <option value="high">Alta (>75%)</option>
                    <option value="medium">Media (50-75%)</option>
                    <option value="low">Baja (<50%)</option>
                </select>
                <select id="time-filter" class="border rounded-lg px-3 py-2">
                    <option value="all">Todo el Día</option>
                    <option value="next-2h">Próximas 2h</option>
                    <option value="next-4h">Próximas 4h</option>
                </select>
                <button id="refresh-btn" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
                    <i data-lucide="refresh-cw" class="w-4 h-4 inline mr-2"></i>
                    Actualizar
                </button>
            </div>
        </div>

        <!-- Matches Grid -->
        <div id="matches-container" class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Matches will be loaded here -->
        </div>

        <!-- Loading State -->
        <div id="loading" class="text-center py-12">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p class="mt-4 text-gray-600">Cargando predicciones...</p>
        </div>
    </div>

    <script>
        // Initialize Lucide icons
        lucide.createIcons();

        // Global state
        let allMatches = [];
        let currentFilters = {
            league: 'all',
            confidence: 'all',
            time: 'all'
        };

        // Load matches on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadMatches();
            setupEventListeners();
        });

        function setupEventListeners() {
            document.getElementById('refresh-btn').addEventListener('click', loadMatches);
            document.getElementById('league-filter').addEventListener('change', applyFilters);
            document.getElementById('confidence-filter').addEventListener('change', applyFilters);
            document.getElementById('time-filter').addEventListener('change', applyFilters);
        }

        async function loadMatches() {
            try {
                document.getElementById('loading').style.display = 'block';
                document.getElementById('matches-container').innerHTML = '';

                const response = await fetch('/api/matches/today');
                const result = await response.json();

                if (result.success) {
                    allMatches = result.data;
                    updateStats();
                    updateLeagueFilter();
                    renderMatches(allMatches);
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                console.error('Error loading matches:', error);
                showError('Error cargando partidos: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function updateStats() {
            document.getElementById('total-matches').textContent = allMatches.length;
            
            const highConfidence = allMatches.filter(m => m.confidence > 75).length;
            document.getElementById('high-confidence').textContent = highConfidence;

            const totalValueBets = allMatches.reduce((sum, m) => sum + m.value_bets.length, 0);
            document.getElementById('value-bets').textContent = totalValueBets;

            if (allMatches.length > 0) {
                const nextMatch = allMatches[0];
                const matchTime = new Date(nextMatch.match_time).toLocaleTimeString('es-ES', {hour: '2-digit', minute: '2-digit'});
                document.getElementById('next-match').textContent = `${matchTime}`;
            }
        }

        function updateLeagueFilter() {
            const leagues = [...new Set(allMatches.map(m => m.league))];
            const select = document.getElementById('league-filter');
            
            // Clear existing options except "all"
            select.innerHTML = '<option value="all">Todas las Ligas</option>';
            
            leagues.forEach(league => {
                const option = document.createElement('option');
                option.value = league;
                option.textContent = league;
                select.appendChild(option);
            });
        }

        function renderMatches(matches) {
            const container = document.getElementById('matches-container');
            container.innerHTML = '';

            if (matches.length === 0) {
                container.innerHTML = `
                    <div class="col-span-full text-center py-12">
                        <p class="text-gray-600">No se encontraron partidos con los filtros seleccionados.</p>
                    </div>
                `;
                return;
            }

            matches.forEach(match => {
                const confidenceClass = match.confidence > 75 ? 'confidence-high' : 
                                      match.confidence > 50 ? 'confidence-medium' : 'confidence-low';

                const matchCard = `
                    <div class="bg-white rounded-lg p-6 card-shadow ${confidenceClass}">
                        <div class="flex justify-between items-start mb-4">
                            <div>
                                <h3 class="font-bold text-lg">${match.home_team} vs ${match.away_team}</h3>
                                <p class="text-gray-600 text-sm">${match.league} • ${new Date(match.match_time).toLocaleString('es-ES')}</p>
                            </div>
                            <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-semibold">
                                ${match.confidence}% confianza
                            </span>
                        </div>

                        <div class="grid grid-cols-3 gap-4 mb-4">
                            <div class="text-center">
                                <p class="text-xs text-gray-600">Local</p>
                                <p class="text-lg font-bold text-green-600">${match.predictions_1x2.home_win}%</p>
                            </div>
                            <div class="text-center">
                                <p class="text-xs text-gray-600">Empate</p>
                                <p class="text-lg font-bold text-yellow-600">${match.predictions_1x2.draw}%</p>
                            </div>
                            <div class="text-center">
                                <p class="text-xs text-gray-600">Visitante</p>
                                <p class="text-lg font-bold text-blue-600">${match.predictions_1x2.away_win}%</p>
                            </div>
                        </div>

                        <div class="grid grid-cols-2 gap-4 mb-4 text-sm">
                            <div>
                                <span class="text-gray-600">Corners:</span>
                                <span class="font-semibold">${match.corners_prediction}</span>
                            </div>
                            <div>
                                <span class="text-gray-600">Goals:</span>
                                <span class="font-semibold">${match.goals_prediction}</span>
                            </div>
                        </div>

                        ${match.value_bets.length > 0 ? `
                            <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
                                <p class="text-xs font-semibold text-yellow-800 mb-2">🎯 VALUE BETS</p>
                                ${match.value_bets.slice(0, 2).map(bet => `
                                    <div class="text-xs text-yellow-700">
                                        ${bet.market}: +${bet.value_percentage}% value
                                    </div>
                                `).join('')}
                            </div>
                        ` : ''}

                        <div class="flex justify-between items-center">
                            <p class="text-sm text-gray-600">${match.recommendation}</p>
                            <button class="text-blue-600 hover:text-blue-800 text-sm font-semibold"
                                    onclick="viewDetails(${match.fixture_id})">
                                Ver Análisis →
                            </button>
                        </div>
                    </div>
                `;
                container.innerHTML += matchCard;
            });
        }

        function applyFilters() {
            currentFilters.league = document.getElementById('league-filter').value;
            currentFilters.confidence = document.getElementById('confidence-filter').value;
            currentFilters.time = document.getElementById('time-filter').value;

            let filteredMatches = allMatches;

            // League filter
            if (currentFilters.league !== 'all') {
                filteredMatches = filteredMatches.filter(m => m.league === currentFilters.league);
            }

            // Confidence filter
            if (currentFilters.confidence !== 'all') {
                switch (currentFilters.confidence) {
                    case 'high':
                        filteredMatches = filteredMatches.filter(m => m.confidence > 75);
                        break;
                    case 'medium':
                        filteredMatches = filteredMatches.filter(m => m.confidence >= 50 && m.confidence <= 75);
                        break;
                    case 'low':
                        filteredMatches = filteredMatches.filter(m => m.confidence < 50);
                        break;
                }
            }

            // Time filter
            if (currentFilters.time !== 'all') {
                const now = new Date();
                filteredMatches = filteredMatches.filter(m => {
                    const matchTime = new Date(m.match_time);
                    const hoursDiff = (matchTime - now) / (1000 * 60 * 60);
                    
                    switch (currentFilters.time) {
                        case 'next-2h':
                            return hoursDiff >= 0 && hoursDiff <= 2;
                        case 'next-4h':
                            return hoursDiff >= 0 && hoursDiff <= 4;
                        default:
                            return true;
                    }
                });
            }

            renderMatches(filteredMatches);
        }

        function viewDetails(fixtureId) {
            alert(`Ver detalles del partido ${fixtureId} - Funcionalidad en desarrollo`);
        }

        function showError(message) {
            document.getElementById('matches-container').innerHTML = `
                <div class="col-span-full bg-red-50 border border-red-200 rounded-lg p-4">
                    <p class="text-red-600">${message}</p>
                </div>
            `;
        }
    </script>
</body>
</html>
    """
    return render_template_string(dashboard_html)

if __name__ == '__main__':
    print("🚀 Iniciando Soccer Predictions API...")
    print("📊 Dashboard disponible en: http://localhost:5000")
    print("🔌 API endpoints disponibles en: http://localhost:5000/api/")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
