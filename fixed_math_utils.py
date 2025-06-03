"""
Utilidades matemáticas para operaciones seguras con NumPy.
Este módulo proporciona funciones para evitar advertencias comunes de NumPy
relacionadas con divisiones por cero, grados de libertad y valores NaN.
"""

import numpy as np
import logging
from typing import List, Union, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def safe_mean(data: List[Union[float, int]], default: float = 0.0) -> float:
    """
    Calcula la media de forma segura evitando advertencias de NumPy.
    
    Args:
        data: Lista de valores numéricos
        default: Valor por defecto si no hay datos válidos
        
    Returns:
        Media de los valores o el valor por defecto
    """
    if not data:
        return default
        
    # Filtrar valores no válidos
    filtered_data = [x for x in data if x is not None and not (isinstance(x, float) and np.isnan(x))]
    
    if not filtered_data:
        return default
        
    return float(np.mean(filtered_data))

def safe_std(data: List[Union[float, int]], default: float = 0.0, ddof: int = 1) -> float:
    """
    Calcula la desviación estándar de forma segura evitando advertencias de NumPy.
    
    Args:
        data: Lista de valores numéricos
        default: Valor por defecto si no hay suficientes datos
        ddof: Delta degrees of freedom (1 para estimación de muestra, 0 para población)
        
    Returns:
        Desviación estándar de los valores o el valor por defecto
    """
    if not data:
        return default
        
    # Filtrar valores no válidos
    filtered_data = [x for x in data if x is not None and not (isinstance(x, float) and np.isnan(x))]
    
    # Necesita al menos ddof+1 elementos para calcular la desviación estándar
    if len(filtered_data) <= ddof:
        return default
        
    return float(np.std(filtered_data, ddof=ddof))

def safe_min(data: List[Union[float, int]], default: float = 0.0) -> float:
    """
    Calcula el mínimo de forma segura evitando errores con listas vacías.
    
    Args:
        data: Lista de valores numéricos
        default: Valor por defecto si no hay datos
        
    Returns:
        Mínimo de los valores o el valor por defecto
    """
    if not data:
        return default
        
    # Filtrar valores no válidos
    filtered_data = [x for x in data if x is not None and not (isinstance(x, float) and np.isnan(x))]
    
    if not filtered_data:
        return default
        
    return float(min(filtered_data))

def safe_max(data: List[Union[float, int]], default: float = 0.0) -> float:
    """
    Calcula el máximo de forma segura evitando errores con listas vacías.
    
    Args:
        data: Lista de valores numéricos
        default: Valor por defecto si no hay datos
        
    Returns:
        Máximo de los valores o el valor por defecto
    """
    if not data:
        return default
        
    # Filtrar valores no válidos
    filtered_data = [x for x in data if x is not None and not (isinstance(x, float) and np.isnan(x))]
    
    if not filtered_data:
        return default
        
    return float(max(filtered_data))

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Realiza una división de forma segura, evitando errores de división por cero.
    
    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor por defecto si el denominador es cero
        
    Returns:
        Resultado de la división o el valor por defecto
    """
    if denominator == 0:
        return default
    return float(numerator / denominator)

def safe_array_std(array: np.ndarray, axis: Optional[int] = None, default: float = 0.0, ddof: int = 1) -> Union[float, np.ndarray]:
    """
    Calcula la desviación estándar de un array NumPy de forma segura.
    
    Args:
        array: Array de NumPy
        axis: Eje a lo largo del cual calcular la desviación estándar
        default: Valor por defecto si no hay suficientes datos
        ddof: Delta degrees of freedom (1 para estimación de muestra, 0 para población)
        
    Returns:
        Desviación estándar o el valor por defecto
    """
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
                return float(result)
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
                if hasattr(result, 'shape'):
                    return np.full_like(result, safe_default, dtype=float)
                else:
                    return np.array([safe_default])
    except Exception as e:
        logger.warning(f"Error al calcular desviación estándar: {e}")
        # Asegurar que el valor de retorno es float o ndarray
        if np.isscalar(default):
            # Verificar que default puede convertirse a float
            if isinstance(default, (int, float)):
                return float(default)
            else:
                return 0.0  # Valor seguro si no es convertible
        else:
            try:
                # Intentar convertir a array con dtype=float, capturando posibles errores
                return np.asarray(default, dtype=float)
            except (TypeError, ValueError):
                # Si falla, devolver un array con un valor seguro
                return np.array([0.0])

def clip_value(value: float, min_val: float, max_val: float) -> float:
    """
    Limita un valor dentro de un rango especificado.
    
    Args:
        value: Valor a limitar
        min_val: Valor mínimo permitido
        max_val: Valor máximo permitido
        
    Returns:
        Valor limitado dentro del rango [min_val, max_val]
    """
    return max(min(value, max_val), min_val)

def normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normaliza un valor en el rango [0, 1] según los límites especificados.
    
    Args:
        value: Valor a normalizar
        min_val: Valor mínimo del rango original
        max_val: Valor máximo del rango original
        
    Returns:
        Valor normalizado en el rango [0, 1]
    """
    if max_val == min_val:
        return 0.5  # Valor central si el rango es cero
    
    # Limitar el valor al rango [min_val, max_val]
    clipped = clip_value(value, min_val, max_val)
    
    # Normalizar al rango [0, 1]
    return (clipped - min_val) / (max_val - min_val)
