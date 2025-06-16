import pickle
from pathlib import Path

# Examinar múltiples archivos cache
files = list(Path('cache').glob('*.cache'))
print('ANÁLISIS DE CONTENIDO DE CACHE')
print('=' * 40)

# Revisar primeros 3 archivos
for i, file in enumerate(files[:3]):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    content = data.get('data', {})
    
    print(f'Archivo {i+1}: {file.name[:16]}...')
    print(f'  Home: {content.get("home_team", "N/A")}')
    print(f'  Away: {content.get("away_team", "N/A")}')
    print(f'  Total campos: {len(content)}')
    print(f'  Campos: {list(content.keys())}')
    print(f'  Tiene predicted_home_goals: {"predicted_home_goals" in content}')
    print(f'  Tiene confidence: {"confidence" in content}')
    print()

# Verificar si hay algún archivo con predicciones reales
print('BUSCANDO PREDICCIONES REALES...')
found_predictions = False
for file in files[:10]:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    content = data.get('data', {})
    
    if content.get('predicted_home_goals') is not None and content.get('predicted_home_goals') != 0:
        print(f'¡Encontrada predicción real en {file.name}!')
        print(f'  Pred: {content.get("predicted_home_goals")} - {content.get("predicted_away_goals")}')
        found_predictions = True
        break

if not found_predictions:
    print('❌ NO se encontraron predicciones reales en los primeros 10 archivos')
    print('   Esto indica que el Master Pipeline no está generando predicciones')
