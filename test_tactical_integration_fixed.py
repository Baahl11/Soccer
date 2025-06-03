"""
Script para probar el análisis táctico utilizando la versión corregida.
"""

print("Ejecutando test de análisis táctico...")

from tactical_integration import get_simplified_tactical_analysis
import json

def test_tactical_analysis():
    """
    Verifica que el análisis táctico incluya ambos métodos:
    - historical
    - neural_network
    """
    # Obtener análisis para dos equipos (IDs 1 y 2 para este test)
    analysis = get_simplified_tactical_analysis(1, 2)
    
    # Verificar que el campo analysis_methods existe
    if 'analysis_methods' not in analysis:
        print("ERROR: Campo 'analysis_methods' no encontrado en el análisis táctico")
        return False
    
    methods = analysis['analysis_methods']
    
    # Verificar que incluye ambos métodos
    if 'historical' not in methods:
        print("ERROR: Método histórico no encontrado en analysis_methods")
        return False
        
    if 'neural_network' not in methods:
        print("ERROR: Método de red neuronal no encontrado en analysis_methods")
        return False
    
    # Verificar la estructura de los métodos
    historical = methods['historical']
    neural = methods['neural_network']
    
    if 'description' not in historical or 'confidence' not in historical:
        print("ERROR: Estructura incorrecta en método histórico")
        return False
    
    if 'description' not in neural:
        print("ERROR: Estructura incorrecta en método de red neuronal")
        return False
    
    # Todo correcto
    print("ÉXITO: El análisis táctico incluye correctamente ambos métodos")
    print(f"- Método histórico: {historical['description']}")
    print(f"- Método neural: {neural['description']}")
    
    # Devolver todo el análisis para referencia
    print("\nEstructura completa del análisis táctico:")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    return True

if __name__ == "__main__":
    test_tactical_analysis()
