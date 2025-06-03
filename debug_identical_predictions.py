#!/usr/bin/env python3
"""
Debug Script: Investigar por qué todas las predicciones son idénticas
"""

import logging
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_match_winner import EnhancedPredictionSystem
from match_winner import predict_match_winner
from team_form import get_team_form, get_head_to_head_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_prediction_pipeline():
    """Debug el pipeline de predicción paso a paso."""
    
    print("🔍 DEPURACIÓN: ¿Por qué todas las predicciones son idénticas?")
    print("=" * 60)
    
    # Casos de prueba diferentes
    test_cases = [
        {"home": 33, "away": 40, "league": 39, "desc": "Man United vs Liverpool"},
        {"home": 50, "away": 42, "league": 39, "desc": "Man City vs Arsenal"},
        {"home": 49, "away": 47, "league": 39, "desc": "Chelsea vs Tottenham"},
        {"home": 529, "away": 530, "league": 140, "desc": "Barcelona vs Atlético Madrid"},
    ]
    
    system = EnhancedPredictionSystem()
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📊 CASO {i}: {case['desc']}")
        print(f"   Home: {case['home']}, Away: {case['away']}, Liga: {case['league']}")
        
        # 1. Verificar datos de forma
        print("\n   🔸 Obteniendo datos de forma...")
        home_form = get_team_form(case['home'], case['league'], None)
        away_form = get_team_form(case['away'], case['league'], None)
        h2h = get_head_to_head_analysis(case['home'], case['away'])
        
        print(f"     Home form keys: {list(home_form.keys()) if home_form else 'None'}")
        print(f"     Away form keys: {list(away_form.keys()) if away_form else 'None'}")
        print(f"     H2H keys: {list(h2h.keys()) if h2h else 'None'}")
        
        # 2. Verificar valores xG por defecto
        print("\n   🔸 Verificando valores xG...")
        home_xg = 1.3  # Valor por defecto
        away_xg = 1.1  # Valor por defecto
        print(f"     Home xG: {home_xg} (¿siempre el mismo?)")
        print(f"     Away xG: {away_xg} (¿siempre el mismo?)")
        
        # 3. Llamar predicción base directamente
        print("\n   🔸 Predicción base (predict_match_winner)...")
        base_pred = predict_match_winner(
            home_team_id=case['home'],
            away_team_id=case['away'],
            league_id=case['league'],
            home_xg=home_xg,
            away_xg=away_xg,
            home_form=home_form,
            away_form=away_form,
            h2h=h2h
        )
        
        base_probs = base_pred.get('probabilities', {})
        print(f"     Base: H={base_probs.get('home_win', 0):.1f}%, "
              f"D={base_probs.get('draw', 0):.1f}%, "
              f"A={base_probs.get('away_win', 0):.1f}%")
        
        # 4. Llamar sistema completo
        print("\n   🔸 Predicción completa (EnhancedPredictionSystem)...")
        enhanced_pred = system.predict(
            home_team_id=case['home'],
            away_team_id=case['away'],
            league_id=case['league']
        )
        
        enhanced_probs = enhanced_pred.get('probabilities', {})
        print(f"     Enhanced: H={enhanced_probs.get('home_win', 0):.1f}%, "
              f"D={enhanced_probs.get('draw', 0):.1f}%, "
              f"A={enhanced_probs.get('away_win', 0):.1f}%")
        
        print("-" * 50)
    
    print("\n🎯 ANÁLISIS:")
    print("Si todos los valores xG son idénticos (1.3 y 1.1), entonces:")
    print("- Las predicciones base serán idénticas")
    print("- Las predicciones mejoradas también serán idénticas")
    print("- Necesitamos calcular xG específicos para cada equipo")

def test_xg_calculation_alternatives():
    """Probar alternativas para calcular xG específicos por equipo."""
    
    print("\n\n🧪 PRUEBA: Alternativas para calcular xG específicos")
    print("=" * 60)
    
    # Método 1: Basado en promedio de goles por partido
    def calculate_simple_xg(team_form):
        """Calcular xG simple basado en forma del equipo."""
        if not team_form:
            return 1.2  # Valor por defecto conservador
        
        # Buscar datos de goles promedio
        goals_scored = team_form.get('goals_scored_per_game', 
                                   team_form.get('avg_goals_scored', 1.2))
        return max(0.5, min(3.0, goals_scored))  # Limitar entre 0.5 y 3.0
    
    # Método 2: Basado en calidad ofensiva vs defensiva
    def calculate_contextual_xg(home_form, away_form, is_home=True):
        """Calcular xG considerando calidad ofensiva vs defensiva."""
        if not home_form or not away_form:
            return 1.3 if is_home else 1.1
        
        if is_home:
            # xG del equipo local = su ataque vs defensa del visitante
            attack = home_form.get('goals_scored_per_game', 1.2)
            opponent_defense = away_form.get('goals_conceded_per_game', 1.2)
        else:
            # xG del equipo visitante = su ataque vs defensa del local
            attack = away_form.get('goals_scored_per_game', 1.1)
            opponent_defense = home_form.get('goals_conceded_per_game', 1.2)
        
        # Combinar ataque propio y debilidad defensiva del rival
        xg = (attack * 0.6) + (opponent_defense * 0.4)
        return max(0.3, min(4.0, xg))
    
    # Probar con casos reales
    test_cases = [
        {"home": 33, "away": 40, "league": 39, "desc": "Man United vs Liverpool"},
        {"home": 50, "away": 42, "league": 39, "desc": "Man City vs Arsenal"},
    ]
    
    for case in test_cases:
        print(f"\n📊 {case['desc']}")
        
        home_form = get_team_form(case['home'], case['league'], None)
        away_form = get_team_form(case['away'], case['league'], None)
        
        # Método 1: Simple
        home_xg_simple = calculate_simple_xg(home_form)
        away_xg_simple = calculate_simple_xg(away_form)
        
        # Método 2: Contextual
        home_xg_contextual = calculate_contextual_xg(home_form, away_form, True)
        away_xg_contextual = calculate_contextual_xg(home_form, away_form, False)
        
        print(f"   Simple xG: Home={home_xg_simple:.2f}, Away={away_xg_simple:.2f}")
        print(f"   Contextual xG: Home={home_xg_contextual:.2f}, Away={away_xg_contextual:.2f}")
        print(f"   Actual (fijo): Home=1.30, Away=1.10")

if __name__ == "__main__":
    debug_prediction_pipeline()
    test_xg_calculation_alternatives()
