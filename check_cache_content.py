#!/usr/bin/env python3
"""
Script para verificar el contenido del cache y determinar si hay datos útiles.
"""

import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import os

def check_cache_files():
    """Verifica el contenido de los archivos de cache."""
    
    cache_dir = Path("cache")
    if not cache_dir.exists():
        print("❌ No existe directorio de cache")
        return
    
    cache_files = list(cache_dir.glob("*.cache"))
    print(f"📂 Encontrados {len(cache_files)} archivos de cache")
    
    current_time = datetime.now().timestamp()
    valid_matches = []
    expired_files = 0
    invalid_files = 0
    
    for i, cache_file in enumerate(cache_files[:10]):  # Solo revisar primeros 10
        try:
            print(f"\n🔍 Revisando {cache_file.name[:16]}...")
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verificar si está expirado (24 horas = 86400 segundos)
            cache_time = cache_data.get('timestamp', 0)
            age_hours = (current_time - cache_time) / 3600
            
            data = cache_data.get('data')
            
            print(f"  ⏰ Edad: {age_hours:.1f} horas")
            
            if age_hours > 24:
                print(f"  ❌ EXPIRADO (>24h)")
                expired_files += 1
                continue
            
            # Verificar tipo de datos
            if isinstance(data, dict):
                # Podría ser un partido individual
                if 'home_team' in data and 'away_team' in data:
                    print(f"  ✅ Partido cached: {data.get('home_team', 'Unknown')} vs {data.get('away_team', 'Unknown')}")
                    valid_matches.append(data)
                elif 'method' in data and data.get('method') == 'discover_matches':
                    print(f"  📊 Cache de descubrimiento de partidos")
                else:
                    print(f"  ❓ Datos tipo dict desconocido")
                    
            elif isinstance(data, list):
                # Podría ser lista de partidos
                if data and isinstance(data[0], dict) and 'home_team' in data[0]:
                    print(f"  ✅ Lista de {len(data)} partidos cached")
                    valid_matches.extend(data)
                else:
                    print(f"  ❓ Lista de {len(data)} elementos desconocidos")
            else:
                print(f"  ❓ Tipo de dato desconocido: {type(data)}")
                
        except Exception as e:
            print(f"  ❌ Error leyendo cache: {e}")
            invalid_files += 1
    
    print(f"\n📊 RESUMEN:")
    print(f"   📁 Total archivos: {len(cache_files)}")
    print(f"   ✅ Partidos válidos encontrados: {len(valid_matches)}")
    print(f"   ❌ Archivos expirados: {expired_files}")
    print(f"   💥 Archivos inválidos: {invalid_files}")
    
    if valid_matches:
        print(f"\n🏆 EJEMPLOS DE PARTIDOS EN CACHE:")
        for i, match in enumerate(valid_matches[:5]):
            print(f"  {i+1}. {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')}")
            print(f"     Liga: {match.get('league', {}).get('name', 'Unknown')}")
            print(f"     Fecha: {match.get('date', 'Unknown')}")
    
    return len(valid_matches) > 0

if __name__ == "__main__":
    print("🔍 VERIFICACIÓN DE CACHE")
    print("=" * 50)
    
    has_valid_data = check_cache_files()
    
    if has_valid_data:
        print(f"\n✅ ¡Hay datos válidos en cache! Puedes usar el sistema sin gastar API.")
    else:
        print(f"\n❌ No hay datos válidos en cache. Necesitarás esperar a que se renueve tu cuota de API.")
