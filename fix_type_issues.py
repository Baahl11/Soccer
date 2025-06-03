"""
Script to fix type-related errors in math_utils.py and stats_utils.py
"""

import os
import shutil
import re
from typing import Dict, Any

def backup_file(filepath: str) -> None:
    """Create a backup of a file with .bak extension if it doesn't already exist."""
    backup_path = f"{filepath}.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        print(f"Created backup: {backup_path}")

def fix_math_utils() -> None:
    """Fix issues in math_utils.py"""
    filepath = "math_utils.py"
    backup_file(filepath)
    
    # Read the current content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Correct the safe_array_std function signature to include ddof parameter
    content = re.sub(
        r"def safe_array_std\(array: np\.ndarray, axis: Optional\[int\] = None, default: float = 0\.0\) -> Union\[float, np\.ndarray\]:",
        "def safe_array_std(array: np.ndarray, axis: Optional[int] = None, default: float = 0.0, ddof: int = 1) -> Union[float, np.ndarray]:",
        content
    )
    
    # Fix 2: Fix the try indentation error
    content = re.sub(
        r'"""    try:',
        '"""\n    try:',
        content
    )
    
    # Fix 3: Add ddof parameter to np.std call
    content = re.sub(
        r"result = np\.std\(array, axis=axis\)",
        "result = np.std(array, axis=axis, ddof=ddof)",
        content
    )
    
    # Fix 4: Replace the hasattr check with safer instanceof checks
    content = re.sub(
        r"isinstance\(default, \(int, float\)\) or hasattr\(default, \"__float__\"\)",
        "isinstance(default, (int, float))",
        content
    )
    
    content = re.sub(
        r"isinstance\(result, \(int, float\)\) or hasattr\(result, \"__float__\"\)",
        "isinstance(result, (int, float))",
        content
    )
    
    # Fix 5: Replace the isnan check with a safer version
    content = re.sub(
        r"if np\.isnan\(result\) if hasattr\(result, \"__float__\"\) else False:",
        "try:\n                if np.isnan(result):",
        content
    )
    
    # Fix 6: Add proper exception handling for isnan
    content = re.sub(
        r"else:\n                    return 0\.0  # Valor seguro por defecto",
        "else:\n                        return 0.0  # Valor seguro por defecto\n                return float(result)\n            except (TypeError, ValueError):\n                # Si result no es un tipo que soporte np.isnan\n                return float(default) if isinstance(default, (int, float)) else 0.0",
        content
    )
    
    # Fix 7: Add safe handling for complex numbers
    content = content.replace(
        "# Convertir a array y forzar tipo float, manejando posibles valores complejos\n            return np.asarray(result, dtype=float)",
        "# Convertir a array y forzar tipo float, manejando posibles valores complejos\n            try:\n                return np.asarray(result, dtype=float)\n            except (TypeError, ValueError):\n                # Si contiene complejos u otros tipos no convertibles a float\n                safe_default = float(default) if isinstance(default, (int, float)) else 0.0\n                if hasattr(result, 'shape'):\n                    return np.full(result.shape, safe_default, dtype=float)\n                else:\n                    return np.array([safe_default])"
    )
    
    # Write the updated content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

def fix_stats_utils() -> None:
    """Fix issues in stats_utils.py"""
    filepath = "stats_utils.py"
    backup_file(filepath)
    
    # Read the current content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Replace hasattr checks with safer isinstance checks
    content = re.sub(
        r"isinstance\(default, \(int, float\)\) or hasattr\(default, \"__float__\"\)",
        "isinstance(default, (int, float))",
        content
    )
    
    content = re.sub(
        r"isinstance\(result, \(int, float\)\) or hasattr\(result, \"__float__\"\)",
        "isinstance(result, (int, float))",
        content
    )
    
    # Fix 2: Fix the isnan check
    content = re.sub(
        r"if np\.isnan\(result\) if hasattr\(result, \"__float__\"\) else False:",
        "try:\n                if np.isnan(result):",
        content
    )
    
    # Fix 3: Fix indentation in the safe_array_std function for the else statement
    content = re.sub(
        r"else:\n                    return 0\.0  # Valor seguro por defecto",
        "else:\n                        return 0.0  # Valor seguro por defecto\n                return float(result)\n            except (TypeError, ValueError):\n                # Si result no es un tipo que soporte np.isnan\n                return float(default) if isinstance(default, (int, float)) else 0.0",
        content
    )
    
    # Fix 4: Add safer array conversion code
    content = content.replace(
        "# Convertir a array y forzar tipo float, manejando posibles valores complejos\n            return np.asarray(result, dtype=float)",
        "# Convertir a array y forzar tipo float, manejando posibles valores complejos\n            try:\n                return np.asarray(result, dtype=float)\n            except (TypeError, ValueError):\n                # Si contiene complejos u otros tipos no convertibles a float\n                safe_default = float(default) if isinstance(default, (int, float)) else 0.0\n                if hasattr(result, 'shape'):\n                    return np.full(result.shape, safe_default, dtype=float)\n                else:\n                    return np.array([safe_default])"
    )
    
    # Write the updated content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

def create_troubleshooting_doc() -> None:
    """Create a troubleshooting document for NumPy warnings and type issues."""
    filepath = "troubleshooting.md"
    
    content = """# Troubleshooting NumPy Warnings and Type Issues

Este documento proporciona información sobre cómo manejar advertencias comunes de NumPy y problemas de tipo en cálculos numéricos.

## Advertencias Comunes de NumPy

### 1. `RuntimeWarning: invalid value encountered in...`

Esta advertencia ocurre cuando NumPy intenta realizar operaciones matemáticas con valores no válidos, como NaN (Not a Number) o Inf (Infinito).

#### Causas comunes:
- División por cero
- Raíz cuadrada de números negativos
- Logaritmo de cero o números negativos
- Operaciones con valores NaN existentes

#### Soluciones:
```python
# Usar np.nan_to_num para reemplazar valores problemáticos
result = np.nan_to_num(array, nan=0.0, posinf=large_value, neginf=-large_value)

# Verificar valores antes de operaciones peligrosas
if denominator != 0:
    result = numerator / denominator
else:
    result = default_value
```

### 2. `RuntimeWarning: Degrees of freedom <= 0 for slice`

Esta advertencia ocurre cuando se calcula la desviación estándar o varianza con un número insuficiente de grados de libertad.

#### Causas comunes:
- Calcular la desviación estándar de un único valor
- Usar `ddof=1` (para estimación muestral) cuando solo hay un valor
- Calcular la desviación estándar a lo largo de un eje con tamaño 1

#### Soluciones:
```python
# Verificar la dimensión antes de calcular
if len(data) > ddof:
    result = np.std(data, ddof=ddof)
else:
    result = default_value

# Cuando se trabaja con ejes específicos
if array.shape[axis] > ddof:
    result = np.std(array, axis=axis, ddof=ddof)
else:
    result = default_value
```

### 3. `TypeError: unsupported operand type` o `ValueError: cannot convert complex to float`

Estos errores ocurren cuando se intenta realizar operaciones con tipos incompatibles, como números complejos cuando se espera un número real.

#### Soluciones:
```python
# Verificar el tipo antes de la conversión
if isinstance(value, (int, float)) and not isinstance(value, complex):
    result = float(value)
else:
    result = default_value

# Usar try-except para capturar errores de conversión
try:
    result = float(value)
except (TypeError, ValueError):
    result = default_value
```

## Prácticas Recomendadas para Cálculos Numéricos Seguros

1. **Siempre proporciona valores predeterminados** para casos de error
2. **Verifica las dimensiones** de los arrays antes de realizar cálculos
3. **Maneja específicamente valores NaN e Inf**:
   ```python
   # Detectar NaN
   if np.isnan(value):
       value = default
   
   # Detectar Inf
   if np.isinf(value):
       value = default
   ```
4. **Usa try-except** para capturar errores de tipo:
   ```python
   try:
       result = operation_that_might_fail(data)
   except (TypeError, ValueError) as e:
       logger.warning(f"Error en operación: {e}")
       result = fallback_value
   ```
5. **Valida los tipos de entrada** antes de realizar operaciones

## Funciones Seguras

El proyecto incluye varias funciones seguras en los módulos `math_utils.py` y `stats_utils.py` que implementan estas prácticas:

- `safe_mean`: Calcula la media de forma segura
- `safe_std`: Calcula la desviación estándar de forma segura
- `safe_divide`: Realiza divisiones evitando errores por división por cero
- `safe_array_std`: Calcula la desviación estándar de arrays NumPy con manejo de errores

## Depuración de Errores Numéricos

Para depurar problemas numéricos, estas herramientas son útiles:

```python
# Activar todas las advertencias de NumPy
np.seterr(all='warn')  

# Ver el contenido detallado de un array
print(np.info(array))

# Comprobar si hay NaN o Inf
print("Contiene NaN:", np.isnan(array).any())
print("Contiene Inf:", np.isinf(array).any())
```"""

    # Don't overwrite an existing file
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created {filepath}")
    else:
        print(f"{filepath} already exists. Not overwriting.")

def update_documentation() -> None:
    """Add reference to troubleshooting.md in market_integration_documentation.md"""
    filepath = "market_integration_documentation.md"
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} does not exist. Cannot update documentation.")
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "troubleshooting.md" not in content:
        # Add reference to the troubleshooting doc
        reference = "\n\n## Referencias Adicionales\n\n- [Solución de problemas con NumPy y errores de tipo](troubleshooting.md): Guía para manejar advertencias comunes de NumPy y problemas de tipos."
        content += reference
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filepath} with reference to troubleshooting doc")
    else:
        print(f"{filepath} already contains reference to troubleshooting. Not modifying.")

if __name__ == "__main__":
    print("Starting fix script...")
    fix_math_utils()
    fix_stats_utils()
    create_troubleshooting_doc()
    update_documentation()
    print("All fixes applied successfully!")
