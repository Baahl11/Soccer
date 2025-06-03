"""
Fix for stats_utils.py
"""

import os

new_content = """
\"\"\"
Utilidades para cálculos estadísticos seguros.
Este módulo proporciona funciones que evitan advertencias comunes de NumPy
relacionadas con divisiones por cero, grados de libertad insuficientes y valores NaN.
\"\"\"

import numpy as np
import logging
from typing import List, Union, Any, Optional, Tuple, Sequence

logger = logging.getLogger(__name__)

def safe_mean(data: Sequence[Union[float, int]], default: float = 0.0) -> float:
    \"\"\"
    Calcula la media de forma segura evitando advertencias de NumPy.
    
    Args:
        data: Secuencia de valores numéricos
        default: Valor por defecto si no hay datos válidos
        
    Returns:
        Media de los valores o el valor por defecto
    \"\"\"
    if not data:
        return default
        
    # Filtrar valores no válidos
    filtered_data = [x for x in data if x is not None and not (isinstance(x, float) and np.isnan(x))]
    
    if not filtered_data:
        return default
        
    return float(np.mean(filtered_data))

def safe_std(data: Sequence[Union[float, int]], default: float = 0.0, ddof: int = 1) -> float:
    \"\"\"
    Calcula la desviación estándar de forma segura evitando advertencias de NumPy.
    
    Args:
        data: Secuencia de valores numéricos
        default: Valor por defecto si no hay suficientes datos
        ddof: Delta degrees of freedom (1 para estimación de muestra, 0 para población)
        
    Returns:
        Desviación estándar de los valores o el valor por defecto
    \"\"\"
    if not data:
        return default
        
    # Filtrar valores no válidos
    filtered_data = [x for x in data if x is not None and not (isinstance(x, float) and np.isnan(x))]
    
    # Necesita al menos ddof+1 elementos para calcular la desviación estándar
    if len(filtered_data) <= ddof:
        return default
        
    return float(np.std(filtered_data, ddof=ddof))

def safe_array_std(array: np.ndarray, axis: Optional[int] = None, 
                  default: float = 0.0, ddof: int = 1) -> Union[float, np.ndarray]:
    \"\"\"
    Calcula la desviación estándar de un array NumPy de forma segura.
    
    Args:
        array: Array de NumPy
        axis: Eje a lo largo del cual calcular la desviación estándar
        default: Valor por defecto si no hay suficientes datos
        ddof: Delta degrees of freedom
        
    Returns:
        Desviación estándar o el valor por defecto
    \"\"\"
    try:
        # Convertir a array NumPy si no lo es ya
        if not isinstance(array, np.ndarray):
            try:
                array = np.asarray(array, dtype=float)
            except (TypeError, ValueError):
                # Si no se puede convertir, devolver un valor seguro
                return default if np.isscalar(default) else np.array([default])
                
        if axis is not None:
            # Verificar que hay suficientes elementos en la dimensión especificada
            shape = array.shape
            if axis < len(shape) and shape[axis] <= ddof:
                # Asegurar que el valor de retorno es float o ndarray
                if np.isscalar(default):
                    # Verificar que default puede convertirse a float
                    if isinstance(default, (int, float)):
                        return float(default)
                    else:
                        return 0.0  # Valor seguro si no es convertible
                else:
                    try:
                        # Si default es un array, asegurar que sea un ndarray con dtype=float
                        return np.asarray(default, dtype=float)
                    except (TypeError, ValueError):
                        # Si falla, devolver un array con un valor seguro
                        return np.array([0.0])
        
        # Intentar hacer el cálculo con manejo de errores para tipos no compatibles
        try:
            result = np.std(array, axis=axis, ddof=ddof)
        except (TypeError, ValueError):
            # Si falla porque los datos contienen tipos no compatibles 
            # (como complejos o cadenas de texto), devolver un valor seguro
            if np.isscalar(default):
                return float(default) if isinstance(default, (int, float)) else 0.0
            else:
                return np.array([0.0])
                
        # Reemplazar valores NaN por el valor predeterminado
        if np.isscalar(result):
            try:
                if np.isnan(result):
                    # Asegurarnos de que default es convertible a float
                    if isinstance(default, (int, float)):
                        return float(default)
                    else:
                        return 0.0  # Valor seguro por defecto
                
                # Verificar que result es convertible a float
                if isinstance(result, (int, float)) and not isinstance(result, complex):
                    return float(result)
                else:
                    return 0.0  # Valor seguro si no es convertible
            except (TypeError, ValueError):
                # Si result no es un tipo que soporte np.isnan
                return float(default) if isinstance(default, (int, float)) else 0.0
        else:
            # Usar un valor seguro para nan que sea convertible a float
            safe_default = float(default) if isinstance(default, (int, float)) else 0.0
            try:
                # Manejar posibles errores en nan_to_num con tipos incompatibles
                result = np.nan_to_num(result, nan=safe_default)
                # Convertir a array y forzar tipo float, manejando posibles valores complejos
                return np.asarray(result, dtype=float)
            except (TypeError, ValueError):
                # Si falla, devolver un array seguro
                if hasattr(result, \"shape\"):
                    return np.full_like(result, safe_default, dtype=float)
                else:
                    return np.array([safe_default])
                    
    except Exception as e:
        logger.warning(f\"Error al calcular desviación estándar: {e}\")
        # Retornar un valor seguro
        if np.isscalar(default):
            if isinstance(default, (int, float)):
                return float(default)
            else:
                return 0.0
        else:
            try:
                return np.asarray(default, dtype=float)
            except (TypeError, ValueError):
                return np.array([0.0])

def safe_array_mean(array: np.ndarray, axis: Optional[int] = None, 
                   default: float = 0.0) -> Union[float, np.ndarray]:
    \"\"\"
    Calcula la media de un array NumPy de forma segura.
    
    Args:
        array: Array de NumPy
        axis: Eje a lo largo del cual calcular la media
        default: Valor por defecto si no hay datos o son inválidos
        
    Returns:
        Media o el valor por defecto
    \"\"\"
    try:
        if isinstance(array, list):
            try:
                array = np.array(array, dtype=float)
            except (TypeError, ValueError):
                return default
            
        if array.size == 0:
            return default
            
        # Calcular la media
        try:
            result = np.mean(array, axis=axis)
        except (TypeError, ValueError):
            # Si falla, devolver un valor seguro
            return default
        
        # Reemplazar valores NaN por el valor predeterminado
        if np.isscalar(result):
            try:
                if np.isnan(result):
                    return default
                
                # Verificar que result es convertible a float
                if isinstance(result, (int, float)) and not isinstance(result, complex):
                    return float(result)
                else:
                    return default
            except (TypeError, ValueError):
                return default
        else:
            try:
                safe_result = np.nan_to_num(result, nan=default)
                return np.asarray(safe_result, dtype=float)
            except (TypeError, ValueError):
                if hasattr(result, \"shape\"):
                    return np.full_like(result, default, dtype=float)
                else:
                    return np.array([default])
        
    except Exception as e:
        logger.warning(f\"Error al calcular media: {e}\")
        return default
"""

# Write to stats_utils.py
with open("stats_utils.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("Archivo stats_utils.py creado correctamente.")
