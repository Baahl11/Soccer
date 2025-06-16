#!/usr/bin/env python3
"""
REVISOR COMPLETO DE PREDICCIONES CACHEADAS
===========================================
Este script revisa TODAS las 262 predicciones cacheadas con detalles completos.
Muestra equipos, ligas, predicciones, confianza, y toda la información disponible.
"""

import pickle
import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def revisar_predicciones_completas():
    """Revisa todas las predicciones cacheadas con detalles completos"""
    
    cache_dir = Path('cache')
    cache_files = list(cache_dir.glob('*.cache'))
    
    print("🔍 REVISOR COMPLETO DE PREDICCIONES CACHEADAS")
    print("=" * 60)
    print(f"📁 Cache Directory: {cache_dir}")
    print(f"📊 Total Cache Files: {len(cache_files)}")
    print()
    
    # Contadores
    predicciones_individuales = []
    discovery_caches = []
    otros_caches = []
    
    # Variables para estadísticas
    leagues_stats = defaultdict(int)
    confidence_levels = []
    team_names = set()
    
    print("📋 ANALIZANDO TODAS LAS PREDICCIONES...")
    print("-" * 60)
    
    for i, cache_file in enumerate(cache_files):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            content = data.get('data')
            timestamp = data.get('timestamp', 0)
            age_hours = (datetime.now().timestamp() - timestamp) / 3600
            
            # Verificar si es una predicción individual
            if isinstance(content, dict) and 'home_team' in content and 'away_team' in content:
                predicciones_individuales.append({
                    'cache_file': cache_file.name,
                    'data': content,
                    'age_hours': age_hours,
                    'timestamp': timestamp
                })
                
                # Recopilar estadísticas
                home_team = content.get('home_team', 'Unknown')
                away_team = content.get('away_team', 'Unknown')
                league_name = content.get('league', {}).get('name', 'Unknown League') if isinstance(content.get('league'), dict) else str(content.get('league', 'Unknown'))
                confidence = content.get('confidence', 0)
                
                team_names.add(home_team)
                team_names.add(away_team)
                leagues_stats[league_name] += 1
                confidence_levels.append(confidence)
                
            elif isinstance(content, list):
                discovery_caches.append({
                    'cache_file': cache_file.name,
                    'matches_count': len(content),
                    'age_hours': age_hours
                })
            else:
                otros_caches.append({
                    'cache_file': cache_file.name,
                    'type': type(content).__name__,
                    'age_hours': age_hours
                })
        
        except Exception as e:
            print(f"❌ Error leyendo {cache_file.name}: {e}")
            continue
    
    # MOSTRAR ESTADÍSTICAS GENERALES
    print(f"📊 ESTADÍSTICAS GENERALES:")
    print(f"   ⚽ Predicciones individuales: {len(predicciones_individuales)}")
    print(f"   🔍 Discovery caches: {len(discovery_caches)}")
    print(f"   📦 Otros caches: {len(otros_caches)}")
    print(f"   🏟️  Equipos únicos: {len(team_names)}")
    print(f"   🏆 Ligas cubiertas: {len(leagues_stats)}")
    
    if confidence_levels:
        avg_confidence = sum(confidence_levels) / len(confidence_levels)
        print(f"   📈 Confianza promedio: {avg_confidence:.1%}")
    
    print()
    
    # MOSTRAR PREDICCIONES COMPLETAS (primeras 20)
    print("🏆 PREDICCIONES COMPLETAS (Primeras 20):")
    print("=" * 60)
    
    # Ordenar por confianza (mayor a menor)
    predicciones_individuales.sort(key=lambda x: x['data'].get('confidence', 0), reverse=True)
    
    for i, pred in enumerate(predicciones_individuales[:20], 1):
        data = pred['data']
        
        home_team = data.get('home_team', 'Unknown')
        away_team = data.get('away_team', 'Unknown')
        league_info = data.get('league', {})
        league_name = league_info.get('name', 'Unknown') if isinstance(league_info, dict) else str(league_info)
        
        # Datos básicos
        fixture_id = data.get('fixture_id', 'N/A')
        confidence = data.get('confidence', 0)
        
        # Predicciones de goles
        pred_home = data.get('predicted_home_goals', 0)
        pred_away = data.get('predicted_away_goals', 0)
        total_goals = data.get('total_goals', 0)
        
        # Probabilidades
        home_win_prob = data.get('home_win_prob', 0)
        draw_prob = data.get('draw_prob', 0)
        away_win_prob = data.get('away_win_prob', 0)
        
        # Probabilidades especiales
        prob_over_25 = data.get('prob_over_2_5', 0)
        prob_btts = data.get('prob_btts', 0)
        
        # Datos de corners y tarjetas
        corners = data.get('corners', {})
        cards = data.get('cards', {})
        
        # ELO ratings
        elo_ratings = data.get('elo_ratings', {})
        
        print(f"{i:2d}. {home_team} vs {away_team}")
        print(f"    🏆 Liga: {league_name}")
        print(f"    🆔 Fixture ID: {fixture_id}")
        print(f"    📈 Confianza: {confidence:.1%}")
        print(f"    ⏰ Cache age: {pred['age_hours']:.1f}h")
        print()
        print(f"    ⚽ PREDICCIÓN DE GOLES:")
        print(f"       🏠 Casa: {pred_home:.1f} goles")
        print(f"       ✈️  Visitante: {pred_away:.1f} goles") 
        print(f"       📊 Total: {total_goals:.1f} goles")
        print()
        print(f"    🎯 PROBABILIDADES 1X2:")
        print(f"       🏠 Victoria Casa: {home_win_prob:.1%}")
        print(f"       🤝 Empate: {draw_prob:.1%}")
        print(f"       ✈️  Victoria Visitante: {away_win_prob:.1%}")
        print()
        print(f"    💰 APUESTAS ESPECIALES:")
        print(f"       📈 Over 2.5 goles: {prob_over_25:.1%}")
        print(f"       ⚽ Ambos marcan: {prob_btts:.1%}")
        print()
        
        # Mostrar corners si están disponibles
        if corners:
            total_corners = corners.get('total', 0)
            home_corners = corners.get('home', 0)
            away_corners = corners.get('away', 0)
            print(f"    🚩 CORNERS:")
            print(f"       📊 Total: {total_corners}")
            print(f"       🏠 Casa: {home_corners} | ✈️  Visitante: {away_corners}")
            print()
        
        # Mostrar tarjetas si están disponibles
        if cards:
            total_cards = cards.get('total', 0)
            home_cards = cards.get('home', 0)
            away_cards = cards.get('away', 0)
            print(f"    🟨 TARJETAS:")
            print(f"       📊 Total: {total_cards}")
            print(f"       🏠 Casa: {home_cards} | ✈️  Visitante: {away_cards}")
            print()
        
        # Mostrar ELO si está disponible
        if elo_ratings:
            home_elo = elo_ratings.get('home_elo', 0)
            away_elo = elo_ratings.get('away_elo', 0)
            elo_diff = elo_ratings.get('elo_difference', 0)
            print(f"    📊 ELO RATINGS:")
            print(f"       🏠 Casa: {home_elo} | ✈️  Visitante: {away_elo}")
            print(f"       ⚖️  Diferencia: {elo_diff:+d}")
            print()
        
        # Información adicional
        method = data.get('method', 'unknown')
        enhanced = data.get('enhanced', False)
        data_source = data.get('data_source', 'unknown')
        
        print(f"    🔧 METADATOS:")
        print(f"       🤖 Método: {method}")
        print(f"       ✨ Enhanced: {'Sí' if enhanced else 'No'}")
        print(f"       📡 Fuente: {data_source}")
        
        print("    " + "-" * 50)
        print()
    
    # MOSTRAR TODAS LAS LIGAS ENCONTRADAS
    print(f"\n🌍 TODAS LAS LIGAS CUBIERTAS ({len(leagues_stats)}):")
    print("-" * 60)
    for league, count in sorted(leagues_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"   {league}: {count} partidos")
    
    # MOSTRAR ESTADÍSTICAS DE CONFIANZA
    if confidence_levels:
        print(f"\n📈 DISTRIBUCIÓN DE CONFIANZA:")
        print("-" * 60)
        high_conf = sum(1 for c in confidence_levels if c >= 0.8)
        medium_conf = sum(1 for c in confidence_levels if 0.6 <= c < 0.8)
        low_conf = sum(1 for c in confidence_levels if c < 0.6)
        
        print(f"   🔥 Alta confianza (≥80%): {high_conf} partidos")
        print(f"   📊 Media confianza (60-80%): {medium_conf} partidos")
        print(f"   ⚠️  Baja confianza (<60%): {low_conf} partidos")
    
    # OPCIÓN PARA VER MÁS PREDICCIONES
    print(f"\n💡 OPCIONES ADICIONALES:")
    print(f"   📋 Total predicciones disponibles: {len(predicciones_individuales)}")
    print(f"   👀 Mostrando las primeras 20 (ordenadas por confianza)")
    print(f"   ⚙️  Para ver más, modifica el número en la línea [:20]")
    
    return {
        'total_predictions': len(predicciones_individuales),
        'total_teams': len(team_names),
        'total_leagues': len(leagues_stats),
        'avg_confidence': sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0,
        'predictions_data': predicciones_individuales
    }

def exportar_predicciones_json():
    """Exporta todas las predicciones a un archivo JSON para análisis"""
    
    print("\n💾 EXPORTANDO PREDICCIONES A JSON...")
    
    cache_dir = Path('cache')
    cache_files = list(cache_dir.glob('*.cache'))
    
    all_predictions = []
    
    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            content = data.get('data')
            timestamp = data.get('timestamp', 0)
            
            if isinstance(content, dict) and 'home_team' in content and 'away_team' in content:
                # Simplificar para JSON
                simplified = {
                    'home_team': content.get('home_team'),
                    'away_team': content.get('away_team'),
                    'league': content.get('league', {}).get('name', 'Unknown') if isinstance(content.get('league'), dict) else str(content.get('league', 'Unknown')),
                    'fixture_id': content.get('fixture_id'),
                    'confidence': content.get('confidence', 0),
                    'predicted_home_goals': content.get('predicted_home_goals', 0),
                    'predicted_away_goals': content.get('predicted_away_goals', 0),
                    'total_goals': content.get('total_goals', 0),
                    'home_win_prob': content.get('home_win_prob', 0),
                    'draw_prob': content.get('draw_prob', 0),
                    'away_win_prob': content.get('away_win_prob', 0),
                    'prob_over_2_5': content.get('prob_over_2_5', 0),
                    'prob_btts': content.get('prob_btts', 0),
                    'corners': content.get('corners', {}),
                    'cards': content.get('cards', {}),
                    'elo_ratings': content.get('elo_ratings', {}),
                    'cache_age_hours': (datetime.now().timestamp() - timestamp) / 3600,
                    'method': content.get('method', 'unknown')
                }
                all_predictions.append(simplified)
        
        except Exception:
            continue
    
    # Guardar en JSON
    output_file = 'todas_las_predicciones.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exportadas {len(all_predictions)} predicciones a: {output_file}")
    print(f"📁 Archivo guardado en: {Path(output_file).absolute()}")
    
    return output_file

if __name__ == "__main__":
    print("🚀 INICIANDO REVISIÓN COMPLETA DE PREDICCIONES...")
    print()
    
    # Revisar predicciones
    resultado = revisar_predicciones_completas()
    
    print()
    print("💡 ¿QUIERES EXPORTAR TODAS LAS PREDICCIONES A JSON?")
    print("   Esto te permitirá analizarlas en Excel o cualquier editor JSON")
    print()
    
    # Exportar automáticamente
    exportar_predicciones_json()
    
    print()
    print("🎉 REVISIÓN COMPLETA FINALIZADA!")
    print(f"✅ {resultado['total_predictions']} predicciones analizadas")
    print(f"🏟️  {resultado['total_teams']} equipos únicos")
    print(f"🏆 {resultado['total_leagues']} ligas cubiertas")
    print(f"📈 Confianza promedio: {resultado['avg_confidence']:.1%}")
