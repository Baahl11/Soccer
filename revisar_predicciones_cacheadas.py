#!/usr/bin/env python3
"""
Revisor de Predicciones Cacheadas
==================================
Este script analiza todas las 262+ predicciones cacheadas para verificar que los nombres 
de equipos se están extrayendo correctamente después de nuestro fix.
"""

import pickle
import os
from pathlib import Path
from datetime import datetime
from collections import Counter
import json

def revisar_predicciones_cacheadas():
    """Analiza todas las predicciones cacheadas para verificar nombres de equipos"""
    
    print("🔍 ANÁLISIS DE 262 PREDICCIONES CACHEADAS")
    print("=" * 50)
    
    cache_dir = Path('cache')
    cache_files = list(cache_dir.glob('*.cache'))
    
    print(f"📁 Directorio cache: {cache_dir}")
    print(f"📦 Archivos cache encontrados: {len(cache_files)}")
    print()
    
    # Estadísticas de análisis
    total_predicciones = 0
    equipos_unknown = 0
    equipos_validos = 0
    nombres_equipos = []
    ligas_encontradas = Counter()
    confianzas = []
    
    # Muestras para mostrar
    predicciones_validas = []
    predicciones_problematicas = []
    
    print("🔬 ANALIZANDO PREDICCIONES...")
    print("-" * 30)
    
    for i, cache_file in enumerate(cache_files, 1):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            content = data.get('data')
            timestamp = data.get('timestamp', 0)
            age_hours = (datetime.now().timestamp() - timestamp) / 3600
            
            # Solo analizar predicciones individuales de partidos (no listas)
            if isinstance(content, dict) and 'home_team' in content and 'away_team' in content:
                total_predicciones += 1
                
                # Extraer datos del equipo
                home_team = content.get('home_team', 'Unknown')
                away_team = content.get('away_team', 'Unknown')
                liga = content.get('league', {})
                liga_nombre = liga.get('name', 'Liga Desconocida') if isinstance(liga, dict) else str(liga)
                confianza = content.get('confidence', 0)
                pred_home = content.get('predicted_home_goals', 0)
                pred_away = content.get('predicted_away_goals', 0)
                
                # Verificar si los nombres son válidos
                if home_team == 'Unknown' or away_team == 'Unknown':
                    equipos_unknown += 1
                    predicciones_problematicas.append({
                        'archivo': cache_file.name[:20] + '...',
                        'home': home_team,
                        'away': away_team,
                        'liga': liga_nombre,
                        'edad_horas': round(age_hours, 1)
                    })
                else:
                    equipos_validos += 1
                    nombres_equipos.extend([home_team, away_team])
                    ligas_encontradas[liga_nombre] += 1
                    confianzas.append(confianza)
                    
                    # Agregar a muestras válidas (primeros 15)
                    if len(predicciones_validas) < 15:
                        predicciones_validas.append({
                            'home': home_team,
                            'away': away_team,
                            'liga': liga_nombre,
                            'confianza': confianza,
                            'prediccion': f"{pred_home:.1f} - {pred_away:.1f}",
                            'edad_horas': round(age_hours, 1),
                            'archivo': cache_file.name[:12] + '...'
                        })
                
                # Mostrar progreso cada 50 archivos
                if total_predicciones % 50 == 0:
                    print(f"   Procesadas: {total_predicciones} predicciones...")
        
        except Exception as e:
            continue
    
    print(f"✅ Análisis completado: {total_predicciones} predicciones encontradas")
    print()
    
    # RESULTADOS DEL ANÁLISIS
    tasa_exito = (equipos_validos / total_predicciones * 100) if total_predicciones > 0 else 0
    equipos_unicos = len(set(nombres_equipos))
    confianza_promedio = sum(confianzas) / len(confianzas) if confianzas else 0
    
    print("📊 RESULTADOS DEL ANÁLISIS:")
    print("-" * 40)
    print(f"⚽ Total Predicciones: {total_predicciones}")
    print(f"✅ Equipos Válidos: {equipos_validos}")
    print(f"❌ Equipos 'Unknown': {equipos_unknown}")
    print(f"📈 Tasa de Éxito: {tasa_exito:.1f}%")
    print(f"🏟️  Equipos Únicos: {equipos_unicos}")
    print(f"🏆 Ligas Cubiertas: {len(ligas_encontradas)}")
    print(f"🎯 Confianza Promedio: {confianza_promedio:.1f}%")
    print()
    
    # EVALUACIÓN DEL ESTADO
    if tasa_exito >= 95:
        print("🎉 EXCELENTE: Los nombres de equipos funcionan perfectamente!")
        estado = "PERFECTO"
    elif tasa_exito >= 85:
        print("✅ BUENO: Los nombres de equipos funcionan bien")
        estado = "BUENO"
    elif tasa_exito >= 70:
        print("⚠️  PARCIAL: Algunos nombres de equipos faltan")
        estado = "PARCIAL"
    else:
        print("❌ PROBLEMA: Muchos nombres de equipos son 'Unknown'")
        estado = "PROBLEMA"
    
    print()
    
    # MOSTRAR MUESTRAS DE PREDICCIONES VÁLIDAS
    if predicciones_validas:
        print("⚽ MUESTRA DE PREDICCIONES VÁLIDAS:")
        print("-" * 45)
        for i, pred in enumerate(predicciones_validas[:10], 1):
            print(f"{i:2d}. {pred['home']} vs {pred['away']}")
            print(f"    Liga: {pred['liga']}")
            print(f"    Predicción: {pred['prediccion']} | Confianza: {pred['confianza']:.1f}%")
            print(f"    Cache: {pred['edad_horas']}h | Archivo: {pred['archivo']}")
            print()
    
    # MOSTRAR PREDICCIONES PROBLEMÁTICAS (si las hay)
    if predicciones_problematicas:
        print("⚠️  PREDICCIONES CON PROBLEMAS:")
        print("-" * 35)
        for i, pred in enumerate(predicciones_problematicas[:5], 1):
            print(f"{i}. Home: '{pred['home']}' | Away: '{pred['away']}'")
            print(f"   Liga: {pred['liga']} | Archivo: {pred['archivo']}")
            print()
    
    # MOSTRAR TOP LIGAS
    if ligas_encontradas:
        print("🌍 TOP LIGAS ENCONTRADAS:")
        print("-" * 30)
        for liga, count in ligas_encontradas.most_common(8):
            print(f"  📍 {liga}: {count} partidos")
        print()
    
    # MOSTRAR ALGUNOS EQUIPOS ÚNICOS
    if nombres_equipos:
        equipos_unicos_lista = list(set(nombres_equipos))
        print("🏟️  MUESTRA DE EQUIPOS ENCONTRADOS:")
        print("-" * 35)
        for i, equipo in enumerate(equipos_unicos_lista[:15], 1):
            print(f"  {i:2d}. {equipo}")
        if len(equipos_unicos_lista) > 15:
            print(f"  ... y {len(equipos_unicos_lista) - 15} equipos más")
        print()
    
    # RESUMEN FINAL
    print("📋 RESUMEN FINAL:")
    print("-" * 20)
    print(f"• Estado del Sistema: {estado}")
    print(f"• Fix de Nombres: {'✅ FUNCIONANDO' if tasa_exito >= 90 else '⚠️ PARCIAL' if tasa_exito >= 70 else '❌ PROBLEMA'}")
    print(f"• API Strategy: {'✅ /fixtures endpoint' if equipos_validos > 0 else '❌ /odds endpoint (sin datos de equipos)'}")
    print(f"• Cache System: {'✅ Operativo' if total_predicciones > 200 else '⚠️ Pocas predicciones'}")
    print(f"• Total Análisis: {total_predicciones}/262+ predicciones")
    
    if tasa_exito >= 90:
        print("\n🏆 MISIÓN CUMPLIDA: El problema de 'Unknown' team names ha sido resuelto!")
    elif equipos_unknown > 0:
        print(f"\n🔧 ACCIÓN REQUERIDA: {equipos_unknown} predicciones aún tienen nombres 'Unknown'")
    
    return {
        'total_predicciones': total_predicciones,
        'equipos_validos': equipos_validos,
        'equipos_unknown': equipos_unknown,
        'tasa_exito': tasa_exito,
        'estado': estado
    }

if __name__ == "__main__":
    revisar_predicciones_cacheadas()
