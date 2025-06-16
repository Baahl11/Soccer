import pickle
import os
from pathlib import Path
from datetime import datetime

# Simple script to review cached predictions
cache_dir = Path('cache')
cache_files = list(cache_dir.glob('*.cache'))

print("REVISOR DE PREDICCIONES CACHEADAS")
print("=" * 50)
print(f"Total archivos: {len(cache_files)}")
print()

predictions = []
count = 0

for cache_file in cache_files:
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        content = data.get('data')
        
        if isinstance(content, dict) and 'home_team' in content and 'away_team' in content:
            count += 1
            home = content.get('home_team', 'Unknown')
            away = content.get('away_team', 'Unknown')
            league = content.get('league', {})
            league_name = league.get('name', 'Unknown') if isinstance(league, dict) else str(league)
            confidence = content.get('confidence', 0)
            pred_home = content.get('predicted_home_goals', 0)
            pred_away = content.get('predicted_away_goals', 0)
            
            predictions.append({
                'home': home,
                'away': away,
                'league': league_name,
                'confidence': confidence,
                'pred_home': pred_home,
                'pred_away': pred_away
            })
            
            # Show first 15 predictions with full details
            if count <= 15:
                print(f"{count:2d}. {home} vs {away}")
                print(f"    Liga: {league_name}")
                print(f"    Prediccion: {pred_home:.1f} - {pred_away:.1f}")
                print(f"    Confianza: {confidence:.1%}")
                print()
    
    except Exception as e:
        continue

print(f"TOTAL PREDICCIONES ENCONTRADAS: {len(predictions)}")
print()

# Show confidence distribution
if predictions:
    high_conf = sum(1 for p in predictions if p['confidence'] >= 0.8)
    med_conf = sum(1 for p in predictions if 0.6 <= p['confidence'] < 0.8)
    low_conf = sum(1 for p in predictions if p['confidence'] < 0.6)
    avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
    
    print("DISTRIBUCION DE CONFIANZA:")
    print(f"  Alta (>=80%): {high_conf}")
    print(f"  Media (60-80%): {med_conf}")
    print(f"  Baja (<60%): {low_conf}")
    print(f"  Promedio: {avg_conf:.1%}")
    print()

# Show unique leagues
leagues = set(p['league'] for p in predictions)
print(f"LIGAS CUBIERTAS: {len(leagues)}")
for league in sorted(leagues):
    count = sum(1 for p in predictions if p['league'] == league)
    print(f"  {league}: {count} partidos")

print()
print("MUESTRAS ADICIONALES (ordenadas por confianza):")
predictions.sort(key=lambda x: x['confidence'], reverse=True)
for i, p in enumerate(predictions[15:25], 16):  # Show next 10
    print(f"{i:2d}. {p['home']} vs {p['away']} ({p['league']}) - {p['confidence']:.1%}")
