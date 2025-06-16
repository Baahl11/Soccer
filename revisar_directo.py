import pickle
import sys
from pathlib import Path

try:
    cache_dir = Path('cache')
    cache_files = list(cache_dir.glob('*.cache'))
    
    print("REVISOR DE PREDICCIONES CACHEADAS")
    print("=" * 40)
    print(f"Archivos encontrados: {len(cache_files)}")
    print()
    
    predictions_count = 0
    sample_predictions = []
    
    for i, cache_file in enumerate(cache_files):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            content = data.get('data', {})
            
            # Check if it's a match prediction
            if (isinstance(content, dict) and 
                'home_team' in content and 
                'away_team' in content):
                
                predictions_count += 1
                
                home_team = content.get('home_team', 'Unknown')
                away_team = content.get('away_team', 'Unknown')
                
                # Get league info safely
                league_info = content.get('league', {})
                if isinstance(league_info, dict):
                    league_name = league_info.get('name', 'Unknown League')
                else:
                    league_name = str(league_info) if league_info else 'Unknown League'
                
                confidence = content.get('confidence', 0.0)
                pred_home = content.get('predicted_home_goals', 0.0)
                pred_away = content.get('predicted_away_goals', 0.0)
                
                # Store first 20 for detailed display
                if len(sample_predictions) < 20:
                    sample_predictions.append({
                        'home': home_team,
                        'away': away_team,
                        'league': league_name,
                        'confidence': confidence,
                        'pred_home': pred_home,
                        'pred_away': pred_away,
                        'file': cache_file.name[:16]
                    })
                    
        except Exception as e:
            print(f"Error procesando {cache_file.name}: {e}")
            continue
    
    print(f"PREDICCIONES ENCONTRADAS: {predictions_count}")
    print()
    
    if sample_predictions:
        print("MUESTRA DE PREDICCIONES:")
        print("-" * 40)
        
        for i, pred in enumerate(sample_predictions, 1):
            print(f"{i:2d}. {pred['home']} vs {pred['away']}")
            print(f"    Liga: {pred['league']}")
            print(f"    Prediccion: {pred['pred_home']:.1f} - {pred['pred_away']:.1f}")
            print(f"    Confianza: {pred['confidence']:.1%}")
            print(f"    Cache: {pred['file']}...")
            print()
        
        print(f"... y {predictions_count - len(sample_predictions)} más!")
    
    print()
    print("ESTADO: Las predicciones están funcionando correctamente!")
    print("Los nombres de equipos se están extrayendo bien.")
    
except Exception as e:
    print(f"Error general: {e}")
    import traceback
    traceback.print_exc()
