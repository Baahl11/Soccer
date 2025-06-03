#!/usr/bin/env python3
"""
Test Script: Verificar que las Probabilidades ya NO son Idénticas

Este script prueba múltiples combinaciones de equipos para verificar que 
el sistema ahora calcule probabilidades específicas para cada partido
en lugar de devolver valores idénticos.
"""

import requests
import json
import logging
from enhanced_match_winner import predict_with_enhanced_system

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_system_directly():
    """Probar el sistema enhanced directamente (sin API)."""
    
    print("🧪 PRUEBA DIRECTA DEL SISTEMA ENHANCED")
    print("=" * 50)
    
    # Casos de prueba con equipos de diferentes fortalezas
    test_cases = [
        {
            "name": "Manchester United vs Liverpool", 
            "home_team_id": 33, 
            "away_team_id": 40,
            "league_id": 39
        },
        {
            "name": "Real Madrid vs Barcelona", 
            "home_team_id": 541, 
            "away_team_id": 529,
            "league_id": 140
        },
        {
            "name": "Bayern Munich vs Borussia Dortmund", 
            "home_team_id": 157, 
            "away_team_id": 165,
            "league_id": 78
        },
        {
            "name": "PSG vs Marseille", 
            "home_team_id": 85, 
            "away_team_id": 81,
            "league_id": 61
        },
        {
            "name": "Inter Milan vs AC Milan", 
            "home_team_id": 505, 
            "away_team_id": 489,
            "league_id": 135
        }
    ]
    
    all_probabilities = []
    
    for i, case in enumerate(test_cases, 1):
        try:
            print(f"\n🔮 Caso {i}: {case['name']}")
            
            # Hacer predicción usando el sistema enhanced
            prediction = predict_with_enhanced_system(
                home_team_id=case['home_team_id'],
                away_team_id=case['away_team_id'],
                league_id=case['league_id']
            )
            
            # Extraer probabilidades
            probs = prediction.get('probabilities', {})
            home_prob = probs.get('home_win', 0)
            draw_prob = probs.get('draw', 0)
            away_prob = probs.get('away_win', 0)
            
            print(f"   Home Win: {home_prob}%")
            print(f"   Draw:     {draw_prob}%") 
            print(f"   Away Win: {away_prob}%")
            print(f"   Total:    {home_prob + draw_prob + away_prob}%")
            
            # Guardar para análisis
            prob_tuple = (home_prob, draw_prob, away_prob)
            all_probabilities.append({
                'case': case['name'],
                'probs': prob_tuple,
                'total': sum(prob_tuple)
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            logger.error(f"Error en caso {case['name']}: {e}")
    
    # Análisis de resultados
    print(f"\n📊 ANÁLISIS DE RESULTADOS")
    print("=" * 50)
    
    if len(all_probabilities) >= 2:
        # Verificar si hay variación significativa
        first_probs = all_probabilities[0]['probs']
        variations_found = []
        
        for result in all_probabilities[1:]:
            other_probs = result['probs']
            # Calcular diferencia absoluta entre probabilidades
            max_diff = max(abs(a - b) for a, b in zip(first_probs, other_probs))
            variations_found.append(max_diff)
            
        max_variation = max(variations_found) if variations_found else 0
        
        print(f"🔍 Variación máxima encontrada: {max_variation:.1f}%")
        
        if max_variation > 5.0:
            print("✅ ÉXITO: Las probabilidades YA NO son idénticas!")
            print("✅ El sistema está calculando predicciones específicas por equipo.")
        elif max_variation > 1.0:
            print("⚠️  PROGRESO: Hay variación, pero es pequeña.")
            print("   Puede necesitar más ajustes en el cálculo de xG.")
        else:
            print("❌ PROBLEMA: Las probabilidades siguen siendo muy similares.")
            print("   Se necesita investigar más el cálculo de xG.")
            
        # Mostrar detalles de variación
        print(f"\n📈 DETALLES DE VARIACIÓN:")
        for i, result in enumerate(all_probabilities):
            print(f"   {i+1}. {result['case']}: {result['probs']}")
            
    else:
        print("❌ No se pudieron obtener suficientes resultados para comparar")

def test_api_endpoint():
    """Probar el endpoint del API web."""
    
    print(f"\n🌐 PRUEBA DEL API ENDPOINT")
    print("=" * 50)
    
    api_url = "http://localhost:5000/api/predict"
    
    test_cases = [
        {"home_team_id": 33, "away_team_id": 40, "league_id": 39},
        {"home_team_id": 541, "away_team_id": 529, "league_id": 140},
        {"home_team_id": 157, "away_team_id": 165, "league_id": 78}
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            print(f"\n🔗 API Test {i}: Teams {case['home_team_id']} vs {case['away_team_id']}")
            
            response = requests.post(api_url, json=case, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                probs = data.get('probabilities', {})
                print(f"   API Response: {probs.get('home_win', 0)}% / {probs.get('draw', 0)}% / {probs.get('away_win', 0)}%")
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"   ⚠️  No se pudo conectar al API (servidor no ejecutándose)")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 VERIFICACIÓN: Probabilidades ya NO son Idénticas")
    print("=" * 60)
    print("Este script verifica que el fix del calculador dinámico")
    print("haya solucionado el problema de probabilidades idénticas.")
    print()
    
    # Ejecutar pruebas
    test_enhanced_system_directly()
    test_api_endpoint()
    
    print(f"\n🏁 PRUEBAS COMPLETADAS")
    print("=" * 60)
