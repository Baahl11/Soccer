"""
Módulo de evaluación avanzada para el sistema de predicción de goles.

Este módulo implementa métricas avanzadas de evaluación basadas en las últimas investigaciones
(ArXiv, 2025) que consideran la naturaleza asimétrica del riesgo en predicciones deportivas.

Características principales:
- Implementación de CRPS (Continuous Ranked Probability Score)
- Evaluaciones basadas en utilidad esperada
- Benchmarking automatizado contra mercados de apuestas
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, log_loss
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PredictionEvaluation:
    """Estructura para almacenar métricas de evaluación"""
    match_id: int
    crps_score: float
    utility_score: float
    calibration_score: float
    market_efficiency: float
    sharpness_score: float
    resolution_score: float
    reliability_score: float