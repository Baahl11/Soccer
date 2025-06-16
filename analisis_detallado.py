import pickle
from pathlib import Path

cache_dir = Path('cache')
files = list(cache_dir.glob('*.cache'))

print('ANALISIS DETALLADO DE PREDICCIONES')
print('=' * 50)

# Examinar una prediccion completa
file = files[0]
with open(file, 'rb') as f:
    data = pickle.load(f)
content = data.get('data', {})

print('ESTRUCTURA COMPLETA DE UNA PREDICCION:')
print('-' * 40)
print(f'Archivo: {file.name}')
print(f'Home team: {content.get("home_team", "N/A")}')
print(f'Away team: {content.get("away_team", "N/A")}')
print()

# Mostrar todas las claves disponibles
print('CAMPOS DISPONIBLES:')
for key in sorted(content.keys()):
    value = content[key]
    if isinstance(value, (dict, list)) and len(str(value)) > 100:
        print(f'  {key}: {type(value).__name__} (complejo)')
    else:
        print(f'  {key}: {value}')

print()
print('PROBLEMAS DETECTADOS:')
if content.get('predicted_home_goals', 0) == 0:
    print('  - Predicted goals = 0 (problema con Master Pipeline)')
if content.get('confidence', 0) == 0:
    print('  - Confidence = 0 (problema con calculo de confianza)')

print()
print('DATOS DEL MASTER PIPELINE:')
if 'accuracy_projection' in content:
    print(f'  Accuracy projection: {content["accuracy_projection"]}')
if 'component_analyses' in content:
    print(f'  Component analyses: {type(content["component_analyses"])}')
    
print()
print('ESTRUCTURA DE LEAGUE:')
league = content.get('league', {})
if isinstance(league, dict):
    for k, v in league.items():
        print(f'  league.{k}: {v}')
else:
    print(f'  league: {league} (tipo: {type(league)})')
