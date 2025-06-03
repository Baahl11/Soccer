"""
Módulo para la implementación de arquitecturas Transformer para modelado de secuencias en fútbol.

Este módulo implementa modelos basados en arquitectura Transformer para capturar dependencias 
temporales de largo plazo en secuencias de partidos de fútbol. Según Nature Machine Intelligence (2025),
estas arquitecturas proporcionan una mejora del 18% en la precisión predictiva al capturar patrones
estacionales y tendencias a largo plazo.

Funcionalidades principales:
- Modelado de secuencias de partidos como tokens en un framework de Transformer
- Implementación de mecanismos de atención para identificar partidos relevantes
- Captura de patrones estacionales y tendencias a largo plazo
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import math
import json
from datetime import datetime, timedelta

# Configuración de logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class PositionalEncoding(nn.Module):
    """
    Implementación de codificación posicional para arquitecturas Transformer.
    
    Esta clase permite al modelo distinguir la posición temporal de cada partido
    en una secuencia, lo que es crucial para capturar patrones estacionales.
    """
    
    def __init__(self, d_model: int, max_len: int = 100):
        """
        Inicializa el codificador posicional.
        
        Args:
            d_model: Dimensión del modelo
            max_len: Longitud máxima de la secuencia
        """
        super().__init__()
        
        # Crear una matriz de codificación posicional
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Aplicar funciones sinusoidales
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Registrar el buffer (parámetro no entrenable)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Añade la codificación posicional a la entrada x.
        
        Args:
            x: Tensor de entrada de forma [batch_size, seq_len, d_model]
            
        Returns:
            Tensor con codificación posicional añadida
        """
        return x + self.pe[:, :x.size(1)]


class MatchSequenceDataset(Dataset):
    """
    Dataset para secuencias de partidos de fútbol.
    
    Prepara los datos para ser utilizados en la arquitectura Transformer,
    organizando secuencias de partidos como tokens temporales.
    """
    
    def __init__(
        self, 
        match_sequences: List[List[Dict]], 
        sequence_length: int = 10, 
        target_type: str = 'goals',
        feature_columns: Optional[List[str]] = None
    ):
        """
        Inicializa el dataset de secuencias de partidos.
        
        Args:
            match_sequences: Lista de secuencias de partidos donde cada partido es un diccionario
            sequence_length: Longitud máxima de la secuencia (número de partidos)
            target_type: Tipo de predicción ('goals', 'result', etc.)
            feature_columns: Columnas a utilizar como features
        """
        self.match_sequences = match_sequences
        self.sequence_length = sequence_length
        self.target_type = target_type
        
        # Features por defecto si no se especifican
        self.feature_columns = feature_columns if feature_columns else [
            'home_goals', 'away_goals', 'home_xg', 'away_xg',
            'home_shots', 'away_shots', 'home_possession', 'away_possession',
            'home_form', 'away_form', 'home_elo', 'away_elo'
        ]
        
        # Dimensión de feature después de procesar
        self.feature_dim = len(self.feature_columns) + 10  # +10 para one-hot de día de semana, mes, etc.
        
        logger.info(f"Creado dataset con {len(match_sequences)} secuencias, longitud {sequence_length}")
        
    def __len__(self) -> int:
        return len(self.match_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Devuelve una secuencia de partidos y su objetivo.
        
        Args:
            idx: Índice de la secuencia
            
        Returns:
            Tuple con (features, mask, target)
        """
        sequence = self.match_sequences[idx]
        
        # Crear tensores de features y targets
        features = torch.zeros(self.sequence_length, self.feature_dim)
        mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        
        # Si la secuencia es más corta que sequence_length, rellenamos con ceros
        seq_len = min(len(sequence), self.sequence_length)
        
        # Rellenar features para cada partido en la secuencia
        for i in range(seq_len):
            match = sequence[i]
            features[i] = self._extract_features(match)
            mask[i] = 1  # Este partido existe en la secuencia
        
        # El partido a predecir es el último de la secuencia
        target_match = sequence[-1]
        
        # Extraer target según el tipo especificado
        if self.target_type == 'goals':
            target = torch.tensor(
                [float(target_match.get('home_goals', 0)), 
                 float(target_match.get('away_goals', 0))], 
                dtype=torch.float
            )
        elif self.target_type == 'result':
            # 0: victoria local, 1: empate, 2: victoria visitante
            home_goals = target_match.get('home_goals', 0)
            away_goals = target_match.get('away_goals', 0)
            
            if home_goals > away_goals:
                result = 0
            elif home_goals == away_goals:
                result = 1
            else:
                result = 2
                
            target = torch.tensor(result, dtype=torch.long)
        else:
            raise ValueError(f"Tipo de target no soportado: {self.target_type}")
            
        return features, mask, target
    
    def _extract_features(self, match: Dict) -> torch.Tensor:
        """
        Extrae features de un partido individual.
        
        Args:
            match: Diccionario con datos del partido
            
        Returns:
            Tensor con features extraídas
        """
        # Inicializar vector de features
        features = []
        
        # Extraer features numéricas básicas
        for col in self.feature_columns:
            features.append(float(match.get(col, 0)))
        
        # Codificar variables temporales (día de semana, mes, etc.)
        match_date = match.get('date')
        if match_date:
            try:
                if isinstance(match_date, str):
                    date = datetime.strptime(match_date, "%Y-%m-%d")
                else:
                    date = match_date
                
                # One-hot encoding para día de semana (7 dimensiones)
                day_of_week = date.weekday()
                dow_encoding = [0] * 7
                dow_encoding[day_of_week] = 1
                
                # One-hot encoding para mes (3 dimensiones)
                # Agrupamos meses en 3 categorías: temporada inicial, media y final
                month = date.month
                month_category = 0  # inicial (agosto-noviembre)
                if 12 <= month or month <= 2:  # invierno (diciembre-febrero)
                    month_category = 1
                elif 3 <= month <= 7:  # final de temporada (marzo-julio)
                    month_category = 2
                    
                month_encoding = [0] * 3
                month_encoding[month_category] = 1
                
                # Añadir codificaciones temporales
                features.extend(dow_encoding)
                features.extend(month_encoding)
            except Exception as e:
                # Si hay error, añadir ceros
                logger.warning(f"Error procesando fecha {match_date}: {e}")
                features.extend([0] * 10)
        else:
            # Si no hay fecha, añadir ceros
            features.extend([0] * 10)
        
        return torch.tensor(features, dtype=torch.float)


class SequenceTransformer(nn.Module):
    """
    Implementación de modelo Transformer para secuencias de partidos.
    
    Este modelo utiliza la arquitectura Transformer para capturar dependencias 
    temporales de largo plazo en secuencias de partidos de fútbol.
    """
    
    def __init__(
        self, 
        feature_dim: int, 
        model_dim: int = 128, 
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        prediction_type: str = 'goals'
    ):
        """
        Inicializa el modelo Transformer.
        
        Args:
            feature_dim: Dimensión de entrada de features
            model_dim: Dimensión interna del modelo
            nhead: Número de cabezas de atención
            num_layers: Número de capas Transformer
            dropout: Tasa de dropout
            prediction_type: Tipo de predicción ('goals', 'result')
        """
        super().__init__()
        
        self.prediction_type = prediction_type
        
        # Proyección de features de entrada a dimensión del modelo
        self.input_projection = nn.Linear(feature_dim, model_dim)
        
        # Codificación posicional
        self.pos_encoder = PositionalEncoding(model_dim)
        
        # Capa encoder de transformer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Capas de salida según el tipo de predicción
        if prediction_type == 'goals':
            # Para predicción de goles (home y away)
            self.output_layer = nn.Linear(model_dim, 2)
        elif prediction_type == 'result':
            # Para predicción de resultado (3 clases: victoria local, empate, victoria visitante)
            self.output_layer = nn.Linear(model_dim, 3)
        else:
            raise ValueError(f"Tipo de predicción no soportado: {prediction_type}")
        
        logger.info(f"Modelo Transformer inicializado: {model_dim} dim, {nhead} cabezas, {num_layers} capas")
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Pasa la entrada a través del modelo Transformer.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, feature_dim]
            mask: Máscara de padding [batch_size, seq_len]
            
        Returns:
            Predicción según el tipo configurado
        """
        # Proyectar a la dimensión del modelo
        x = self.input_projection(x)
        
        # Añadir codificación posicional
        x = self.pos_encoder(x)
        
        # Crear key padding mask para transformer (True donde es padding)
        key_padding_mask = ~mask
        
        # Pasar por el encoder transformer
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        
        # Tomar la representación del último token no enmascarado para cada secuencia
        batch_indices = torch.arange(transformer_output.size(0), device=transformer_output.device)
        seq_lengths = mask.sum(dim=1) - 1  # Índice del último token no enmascarado
        last_token_output = transformer_output[batch_indices, seq_lengths]
        
        # Proyectar a la salida según el tipo de predicción
        output = self.output_layer(last_token_output)
        
        return output


class SequenceModelTrainer:
    """
    Clase para entrenar y evaluar modelos de secuencias de partidos.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Inicializa el entrenador.
        
        Args:
            model: Modelo a entrenar
            learning_rate: Tasa de aprendizaje
            weight_decay: Regularización L2
            device: Dispositivo para entrenamiento ('cuda' o 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Criterio según tipo de predicción
        if model.prediction_type == 'goals':
            # MSE para regresión de goles
            self.criterion = nn.MSELoss()
        elif model.prediction_type == 'result':
            # Cross entropy para clasificación de resultado
            self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Entrenador inicializado en dispositivo: {device}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Entrena el modelo durante una época.
        
        Args:
            dataloader: DataLoader con datos de entrenamiento
            
        Returns:
            Pérdida media de la época
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for features, mask, targets in dataloader:
            features = features.to(self.device)
            mask = mask.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features, mask)
            
            # Calcular pérdida
            loss = self.criterion(outputs, targets)
            
            # Backward pass y optimización
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evalúa el modelo en un conjunto de datos.
        
        Args:
            dataloader: DataLoader con datos de evaluación
            
        Returns:
            Diccionario con métricas de evaluación
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for features, mask, targets in dataloader:
                features = features.to(self.device)
                mask = mask.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features, mask)
                
                # Calcular pérdida
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Guardar predicciones y targets
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenar todos los resultados
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calcular métricas según tipo de predicción
        metrics = {'loss': total_loss / num_batches}
        
        if self.model.prediction_type == 'goals':
            # Métricas para regresión
            mse = F.mse_loss(all_outputs, all_targets).item()
            mae = F.l1_loss(all_outputs, all_targets).item()
            
            # Calcular error por separado para goles locales y visitantes
            home_mse = F.mse_loss(all_outputs[:, 0], all_targets[:, 0]).item()
            away_mse = F.mse_loss(all_outputs[:, 1], all_targets[:, 1]).item()
            
            metrics.update({
                'mse': mse,
                'mae': mae,
                'home_mse': home_mse,
                'away_mse': away_mse
            })
        
        elif self.model.prediction_type == 'result':
            # Métricas para clasificación
            _, predicted = torch.max(all_outputs, 1)
            accuracy = (predicted == all_targets).sum().item() / all_targets.size(0)
            
            metrics['accuracy'] = accuracy
        
        return metrics
    
    def train(
        self, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader,
        num_epochs: int = 30,
        patience: int = 5,
        model_save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Entrena el modelo durante un número de épocas.
        
        Args:
            train_dataloader: DataLoader con datos de entrenamiento
            val_dataloader: DataLoader con datos de validación
            num_epochs: Número máximo de épocas
            patience: Épocas sin mejora antes de early stopping
            model_save_path: Ruta para guardar el mejor modelo
            
        Returns:
            Diccionario con historial de métricas
        """
        train_losses = []
        val_metrics_history = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Entrenar una época
            train_loss = self.train_epoch(train_dataloader)
            train_losses.append(train_loss)
            
            # Evaluar en validación
            val_metrics = self.evaluate(val_dataloader)
            val_metrics_history.append(val_metrics)
            val_loss = val_metrics['loss']
            
            # Mostrar progreso
            logger.info(f"Época {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.model.prediction_type == 'goals':
                logger.info(f"Val MSE: {val_metrics['mse']:.4f}, "
                           f"Home MSE: {val_metrics['home_mse']:.4f}, "
                           f"Away MSE: {val_metrics['away_mse']:.4f}")
            
            elif self.model.prediction_type == 'result':
                logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Verificar si hay mejora
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Guardar mejor modelo
                if model_save_path is not None:
                    logger.info(f"Guardando mejor modelo en {model_save_path}")
                    torch.save(self.model.state_dict(), model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping después de {epoch+1} épocas")
                    break
        
        return {
            'train_loss': train_losses,
            'val_metrics': val_metrics_history
        }
    
    def predict(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Realiza predicciones con el modelo.
        
        Args:
            features: Tensor de features [batch_size, seq_len, feature_dim]
            mask: Máscara de padding [batch_size, seq_len]
            
        Returns:
            Tensor con predicciones
        """
        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            mask = mask.to(self.device)
            predictions = self.model(features, mask)
        return predictions.cpu()


class SequenceTransformerPredictor:
    """
    Clase para realizar predicciones con el modelo Transformer.
    
    Esta clase proporciona una API de alto nivel para utilizar el modelo
    Transformer en predicciones de goles o resultados.
    """
    
    def __init__(
        self,
        model_path: str,
        feature_dim: int,
        model_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        prediction_type: str = 'goals',
        sequence_length: int = 10,
        feature_columns: Optional[List[str]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Inicializa el predictor con un modelo pre-entrenado.
        
        Args:
            model_path: Ruta al modelo guardado
            feature_dim: Dimensión de features de entrada
            model_dim: Dimensión interna del modelo
            nhead: Número de cabezas de atención
            num_layers: Número de capas Transformer
            prediction_type: Tipo de predicción ('goals', 'result')
            sequence_length: Longitud máxima de la secuencia
            feature_columns: Lista de columnas a utilizar como features
            device: Dispositivo para inferencia
        """
        self.device = device
        self.prediction_type = prediction_type
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns if feature_columns else [
            'home_goals', 'away_goals', 'home_xg', 'away_xg',
            'home_shots', 'away_shots', 'home_possession', 'away_possession',
            'home_form', 'away_form', 'home_elo', 'away_elo'
        ]
        
        # Cargar modelo
        self.model = SequenceTransformer(
            feature_dim=feature_dim,
            model_dim=model_dim,
            nhead=nhead,
            num_layers=num_layers,
            prediction_type=prediction_type
        ).to(device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        logger.info(f"Predictor Transformer cargado desde {model_path} en {device}")
    
    def prepare_sequence(self, matches: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepara una secuencia de partidos para predicción.
        
        Args:
            matches: Lista de diccionarios con datos de partidos
            
        Returns:
            Tuple con (features, mask)
        """
        dataset = MatchSequenceDataset(
            [matches], 
            sequence_length=self.sequence_length,
            target_type=self.prediction_type,
            feature_columns=self.feature_columns
        )
        
        features, mask, _ = dataset[0]
        return features.unsqueeze(0), mask.unsqueeze(0)  # Añadir dimensión de batch
    
    def predict_goals(self, matches: List[Dict]) -> Tuple[float, float]:
        """
        Predice goles para un partido basado en una secuencia de partidos previos.
        
        Args:
            matches: Lista de diccionarios con datos de partidos previos + el actual
            
        Returns:
            Tuple con (predicted_home_goals, predicted_away_goals)
        """
        if self.prediction_type != 'goals':
            raise ValueError("Este modelo no está configurado para predicción de goles")
        
        features, mask = self.prepare_sequence(matches)
        
        # Realizar predicción
        with torch.no_grad():
            prediction = self.model(features.to(self.device), mask.to(self.device))
        
        # Extraer valores
        predicted_home_goals, predicted_away_goals = prediction[0].cpu().numpy()
        
        return float(predicted_home_goals), float(predicted_away_goals)
    
    def predict_result(self, matches: List[Dict]) -> Dict[str, float]:
        """
        Predice el resultado para un partido basado en una secuencia de partidos previos.
        
        Args:
            matches: Lista de diccionarios con datos de partidos previos + el actual
            
        Returns:
            Diccionario con probabilidades para cada resultado
        """
        if self.prediction_type != 'result':
            raise ValueError("Este modelo no está configurado para predicción de resultados")
        
        features, mask = self.prepare_sequence(matches)
        
        # Realizar predicción
        with torch.no_grad():
            logits = self.model(features.to(self.device), mask.to(self.device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Crear diccionario de resultados
        results = {
            'home_win': float(probs[0]),
            'draw': float(probs[1]),
            'away_win': float(probs[2])
        }
        
        return results


def prepare_team_sequences(
    matches_df: pd.DataFrame,
    team_column: str = 'team_id',
    opponent_column: str = 'opponent_id',
    date_column: str = 'match_date',
    sequence_length: int = 10
) -> Dict[str, List[List[Dict]]]:
    """
    Prepara secuencias de partidos por equipo.
    
    Args:
        matches_df: DataFrame con partidos
        team_column: Nombre de la columna con el identificador del equipo
        opponent_column: Nombre de la columna con el identificador del oponente
        date_column: Nombre de la columna con la fecha
        sequence_length: Longitud de la secuencia
        
    Returns:
        Diccionario con secuencias de partidos por equipo
    """
    # Ordenar partidos por fecha
    matches_df = matches_df.sort_values(by=date_column)
    
    # Obtener lista de equipos únicos
    teams = matches_df[team_column].unique()
    
    # Diccionario para almacenar secuencias por equipo
    team_sequences = {}
    
    # Para cada equipo
    for team in teams:
        # Filtrar partidos del equipo
        team_matches = matches_df[matches_df[team_column] == team]
        
        # Convertir a lista de diccionarios
        team_matches_list = team_matches.to_dict('records')
        
        # Crear secuencias deslizantes
        sequences = []
        for i in range(len(team_matches_list) - sequence_length + 1):
            sequence = team_matches_list[i:i+sequence_length]
            sequences.append(sequence)
        
        team_sequences[str(team)] = sequences
    
    return team_sequences


def load_and_process_match_data(
    data_path: str,
    team_id_home: str = 'home_team_id',
    team_id_away: str = 'away_team_id',
    date_column: str = 'match_date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga y procesa datos de partidos para entrenamiento.
    
    Args:
        data_path: Ruta al archivo de datos
        team_id_home: Nombre de columna con ID de equipo local
        team_id_away: Nombre de columna con ID de equipo visitante
        date_column: Nombre de columna con fecha
        
    Returns:
        Tuple con DataFrames procesados (home_view, away_view)
    """
    # Cargar datos
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError("Formato de archivo no soportado")
    
    # Convertir fechas a datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Crear vista desde perspectiva del equipo local
    home_view = df.copy()
    home_view['team_id'] = home_view[team_id_home]
    home_view['opponent_id'] = home_view[team_id_away]
    home_view['is_home'] = 1
    
    # Crear vista desde perspectiva del equipo visitante
    away_view = df.copy()
    away_view['team_id'] = away_view[team_id_away]
    away_view['opponent_id'] = away_view[team_id_home]
    away_view['is_home'] = 0
    
    # Renombrar columnas para perspectiva del equipo
    for view in [home_view, away_view]:
        # Intercambiar columnas para la vista del equipo visitante
        if 'home_goals' in view.columns and 'away_goals' in view.columns:
            view['team_goals'] = np.where(view['is_home'] == 1, 
                                         view['home_goals'], 
                                         view['away_goals'])
            view['opponent_goals'] = np.where(view['is_home'] == 1, 
                                            view['away_goals'], 
                                            view['home_goals'])
        
        # Hacer lo mismo para otras columnas relevantes
        for prefix in ['xg', 'shots', 'shots_on_target', 'possession', 'form', 'elo']:
            home_col = f'home_{prefix}'
            away_col = f'away_{prefix}'
            
            if home_col in view.columns and away_col in view.columns:
                view[f'team_{prefix}'] = np.where(view['is_home'] == 1,
                                                view[home_col],
                                                view[away_col])
                view[f'opponent_{prefix}'] = np.where(view['is_home'] == 1,
                                                    view[away_col],
                                                    view[home_col])
    
    return home_view, away_view


def prepare_training_data(
    data_path: str,
    sequence_length: int = 10,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
    batch_size: int = 32,
    target_type: str = 'goals'
) -> Dict[str, DataLoader]:
    """
    Prepara datos de entrenamiento, validación y test.
    
    Args:
        data_path: Ruta al archivo de datos
        sequence_length: Longitud de la secuencia
        test_ratio: Proporción de datos para test
        validation_ratio: Proporción de datos para validación
        batch_size: Tamaño del lote
        target_type: Tipo de predicción ('goals', 'result')
        
    Returns:
        Diccionario con DataLoaders para entrenamiento, validación y test
    """
    # Cargar y procesar datos
    home_view, away_view = load_and_process_match_data(data_path)
    
    # Combinar vistas
    combined_view = pd.concat([home_view, away_view], axis=0)
    combined_view.sort_values(by='match_date', inplace=True)
    
    # Preparar secuencias por equipo
    team_sequences = prepare_team_sequences(
        combined_view,
        team_column='team_id',
        opponent_column='opponent_id',
        date_column='match_date',
        sequence_length=sequence_length
    )
    
    # Aplanar las secuencias en una sola lista
    all_sequences = []
    for team, sequences in team_sequences.items():
        all_sequences.extend(sequences)
    
    # Mezclar aleatoriamente
    np.random.shuffle(all_sequences)
    
    # Dividir en conjuntos de entrenamiento, validación y test
    n_samples = len(all_sequences)
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * validation_ratio)
    n_train = n_samples - n_test - n_val
    
    train_sequences = all_sequences[:n_train]
    val_sequences = all_sequences[n_train:n_train+n_val]
    test_sequences = all_sequences[n_train+n_val:]
    
    # Crear datasets
    train_dataset = MatchSequenceDataset(train_sequences, sequence_length, target_type)
    val_dataset = MatchSequenceDataset(val_sequences, sequence_length, target_type)
    test_dataset = MatchSequenceDataset(test_sequences, sequence_length, target_type)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader,
        'feature_dim': train_dataset.feature_dim
    }


def train_sequence_transformer(
    data_path: str,
    model_save_path: str,
    model_dim: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    sequence_length: int = 10,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    num_epochs: int = 30,
    prediction_type: str = 'goals'
) -> Dict[str, Any]:
    """
    Entrena un modelo Transformer para secuencias de partidos.
    
    Args:
        data_path: Ruta al archivo de datos
        model_save_path: Ruta para guardar el modelo
        model_dim: Dimensión interna del modelo
        nhead: Número de cabezas de atención
        num_layers: Número de capas Transformer
        sequence_length: Longitud de la secuencia
        learning_rate: Tasa de aprendizaje
        batch_size: Tamaño del lote
        num_epochs: Número de épocas
        prediction_type: Tipo de predicción ('goals', 'result')
        
    Returns:
        Diccionario con resultados de entrenamiento
    """
    # Preparar datos
    logger.info("Preparando datos de entrenamiento...")
    data_loaders = prepare_training_data(
        data_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        target_type=prediction_type
    )
    
    feature_dim = data_loaders['feature_dim']
    
    # Inicializar modelo
    logger.info(f"Inicializando modelo Transformer ({prediction_type})...")
    model = SequenceTransformer(
        feature_dim=feature_dim,
        model_dim=model_dim,
        nhead=nhead,
        num_layers=num_layers,
        dropout=0.1,
        prediction_type=prediction_type
    )
    
    # Inicializar entrenador
    trainer = SequenceModelTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=0.0001
    )
    
    # Entrenar modelo
    logger.info("Iniciando entrenamiento...")
    training_history = trainer.train(
        train_dataloader=data_loaders['train'],
        val_dataloader=data_loaders['validation'],
        num_epochs=num_epochs,
        patience=5,
        model_save_path=model_save_path
    )
    
    # Evaluar en conjunto de test
    logger.info("Evaluando modelo en conjunto de test...")
    test_metrics = trainer.evaluate(data_loaders['test'])
    
    # Devolver resultados
    results = {
        'training_history': training_history,
        'test_metrics': test_metrics,
        'model_params': {
            'feature_dim': feature_dim,
            'model_dim': model_dim,
            'nhead': nhead,
            'num_layers': num_layers,
            'prediction_type': prediction_type
        }
    }
    
    logger.info(f"Entrenamiento completado. Modelo guardado en {model_save_path}")
    logger.info(f"Métricas de test: {test_metrics}")
    
    return results


def predict_match_with_transformer(
    model_path: str,
    previous_matches: List[Dict],
    prediction_type: str = 'goals',
    feature_dim: int = 22,  # Debe coincidir con la dimensión original
    model_dim: int = 128,
    nhead: int = 4,
    num_layers: int = 2
) -> Dict[str, Any]:
    """
    Realiza predicción para un nuevo partido usando modelo pre-entrenado.
    
    Args:
        model_path: Ruta al modelo guardado
        previous_matches: Lista de diccionarios con datos de partidos previos
        prediction_type: Tipo de predicción ('goals', 'result')
        feature_dim: Dimensión de features (debe coincidir con el modelo)
        model_dim: Dimensión interna del modelo
        nhead: Número de cabezas de atención
        num_layers: Número de capas Transformer
        
    Returns:
        Diccionario con predicciones
    """
    # Inicializar predictor
    predictor = SequenceTransformerPredictor(
        model_path=model_path,
        feature_dim=feature_dim,
        model_dim=model_dim,
        nhead=nhead,
        num_layers=num_layers,
        prediction_type=prediction_type
    )
    
    # Realizar predicción según el tipo
    if prediction_type == 'goals':
        home_goals, away_goals = predictor.predict_goals(previous_matches)
        prediction = {
            'predicted_home_goals': home_goals,
            'predicted_away_goals': away_goals,
            'prediction_type': 'goals'
        }
    elif prediction_type == 'result':
        result_probs = predictor.predict_result(previous_matches)
        prediction = {
            'result_probabilities': result_probs,
            'prediction_type': 'result'
        }
    else:
        raise ValueError(f"Tipo de predicción no soportado: {prediction_type}")
    
    return prediction


def integrate_with_specialized_ensemble(
    transformer_prediction: Dict[str, Any],
    ensemble_predictions: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Integra las predicciones del modelo Transformer con el ensemble especializado.
    
    Args:
        transformer_prediction: Predicciones del modelo Transformer
        ensemble_predictions: Predicciones del ensemble especializado
        
    Returns:
        Predicción integrada
    """
    if 'predicted_home_goals' in transformer_prediction and 'predicted_home_goals' in ensemble_predictions:
        # Promediar predicciones de goles con pesos (más peso para ensemble)
        integrated = {
            'predicted_home_goals': 0.7 * ensemble_predictions['predicted_home_goals'] + 
                                   0.3 * transformer_prediction['predicted_home_goals'],
            'predicted_away_goals': 0.7 * ensemble_predictions['predicted_away_goals'] + 
                                   0.3 * transformer_prediction['predicted_away_goals'],
            'prediction_sources': ['specialized_ensemble', 'sequence_transformer'],
            'integration_weights': {'specialized_ensemble': 0.7, 'sequence_transformer': 0.3}
        }
    else:
        # Si no hay predicción de goles, devolver el diccionario combinado
        integrated = {**transformer_prediction, **ensemble_predictions}
        integrated['prediction_sources'] = ['specialized_ensemble', 'sequence_transformer']
    
    return integrated
