"""
Módulo para implementación de Graph Neural Networks (GNNs) para modelado de relaciones entre equipos.

Este módulo implementa modelos GNN para capturar relaciones complejas entre equipos de fútbol,
basado en investigaciones de KDD 2024 sobre el modelado de deportes como grafos.

Funcionalidades principales:
- Modelar la liga como un grafo donde los equipos son nodos
- Representar partidos como conexiones ponderadas entre nodos
- Capturar "estilos de juego" como propiedades emergentes del grafo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import networkx as nx
import dgl
from dgl.nn import GraphConv, GATConv, SAGEConv
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configuración de logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TeamGraphDataset:
    """
    Dataset que representa una liga como un grafo donde los equipos son nodos
    y los partidos son aristas dirigidas.
    """
    
    def __init__(
        self,
        match_df: pd.DataFrame,
        team_id_home_col: str = 'home_team_id',
        team_id_away_col: str = 'away_team_id',
        result_cols: List[str] = ['home_goals', 'away_goals'],
        feature_cols: Optional[List[str]] = None,
        time_window: Optional[int] = None,
        normalize_features: bool = True
    ):
        """
        Inicializa el dataset de grafo de equipos.
        
        Args:
            match_df: DataFrame con datos de partidos
            team_id_home_col: Nombre de columna con ID del equipo local
            team_id_away_col: Nombre de columna con ID del equipo visitante
            result_cols: Columnas con resultados (goles)
            feature_cols: Columnas adicionales a usar como características
            time_window: Ventana de tiempo para filtrar partidos (en días)
            normalize_features: Si se deben normalizar las características
        """
        self.match_df = match_df.copy()
        self.team_id_home_col = team_id_home_col
        self.team_id_away_col = team_id_away_col
        self.result_cols = result_cols
        self.time_window = time_window
        self.normalize_features = normalize_features
        
        # Configurar features por defecto si no se especifican
        self.feature_cols = feature_cols if feature_cols else [
            'home_xg', 'away_xg', 
            'home_shots', 'away_shots', 
            'home_possession', 'away_possession'
        ]
        
        # Verificar que las columnas existen
        required_cols = [team_id_home_col, team_id_away_col] + result_cols
        missing_cols = [col for col in required_cols if col not in match_df.columns]
        if missing_cols:
            raise ValueError(f"Columnas requeridas no encontradas: {missing_cols}")
        
        # Obtener lista de equipos únicos
        self.teams = pd.concat([
            match_df[team_id_home_col],
            match_df[team_id_away_col]
        ]).unique()
        
        # Mapeo de IDs de equipos a índices
        self.team_to_idx = {team: idx for idx, team in enumerate(self.teams)}
        self.idx_to_team = {idx: team for team, idx in self.team_to_idx.items()}
        
        # Generar grafo
        self.graph = self._build_graph()
        
        logger.info(f"Dataset inicializado con {len(self.teams)} equipos y "
                   f"{len(match_df)} partidos")
    
    def _build_graph(self) -> dgl.DGLGraph:
        """
        Construye un grafo dirigido donde los equipos son nodos y los partidos son aristas.
        
        Returns:
            Grafo DGL construido
        """
        # Aplicar filtro de tiempo si se especifica
        df = self.match_df
        if self.time_window is not None and 'date' in df.columns:
            if isinstance(df['date'].iloc[0], str):
                df['date'] = pd.to_datetime(df['date'])
            cutoff_date = df['date'].max() - timedelta(days=self.time_window)
            df = df[df['date'] >= cutoff_date]
        
        # Crear grafo dirigido
        g = dgl.graph(([], []), num_nodes=len(self.teams))
        
        # Listas para aristas y características
        src_nodes = []
        dst_nodes = []
        edge_features = []
        
        # Calcular features de aristas (partidos)
        for _, row in df.iterrows():
            # Índices de nodos
            home_idx = self.team_to_idx[row[self.team_id_home_col]]
            away_idx = self.team_to_idx[row[self.team_id_away_col]]
            
            # Añadir aristas en ambas direcciones (local a visitante y viceversa)
            src_nodes.extend([home_idx, away_idx])
            dst_nodes.extend([away_idx, home_idx])
            
            # Extraer features para la arista home -> away
            home_to_away_features = []
            for col in self.feature_cols:
                if col in row:
                    home_to_away_features.append(row[col])
                else:
                    home_to_away_features.append(0.0)
            
            # Añadir resultado como feature (goles)
            for col in self.result_cols:
                if col in row:
                    home_to_away_features.append(float(row[col]))
                else:
                    home_to_away_features.append(0.0)
            
            # Crear features para la arista away -> home (invertir algunas métricas)
            away_to_home_features = home_to_away_features.copy()
            
            # Invertir métricas específicas como xg, shots, etc.
            for i, col in enumerate(self.feature_cols):
                if 'home_' in col or 'away_' in col:
                    # Intercambiar métricas home/away
                    away_to_home_features[i] = home_to_away_features[i+1] if i % 2 == 0 else home_to_away_features[i-1]
            
            # Invertir resultados (goles)
            home_goals_idx = len(self.feature_cols)
            away_goals_idx = len(self.feature_cols) + 1
            away_to_home_features[home_goals_idx] = home_to_away_features[away_goals_idx]
            away_to_home_features[away_goals_idx] = home_to_away_features[home_goals_idx]
            
            # Añadir features a la lista
            edge_features.append(home_to_away_features)
            edge_features.append(away_to_home_features)
        
        # Añadir aristas y características al grafo
        g.add_edges(src_nodes, dst_nodes)
        
        # Convertir features a tensor
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        
        # Normalizar features si está habilitado
        if self.normalize_features and edge_features.shape[0] > 0:
            mean = edge_features.mean(dim=0, keepdim=True)
            std = edge_features.std(dim=0, keepdim=True)
            std[std == 0] = 1  # Evitar división por cero
            edge_features = (edge_features - mean) / std
        
        # Añadir características de aristas
        g.edata['features'] = edge_features
        
        # Crear características iniciales de nodos (equipos)
        # Inicialmente son ceros, se actualizarán con información de partidos
        g.ndata['h'] = torch.zeros((g.number_of_nodes(), len(self.feature_cols) + len(self.result_cols)))
        
        # Añadir metadatos adicionales
        g.ndata['team_id'] = torch.tensor([self.idx_to_team[i] for i in range(len(self.teams))])
        
        return g
    
    def get_node_features(self, aggregation: str = 'mean') -> torch.Tensor:
        """
        Genera características para los nodos (equipos) a partir de las aristas (partidos).
        
        Args:
            aggregation: Método de agregación ('mean', 'sum', 'max')
            
        Returns:
            Tensor con características de nodos
        """
        g = self.graph
        
        # Inicializar características de nodos
        num_nodes = g.number_of_nodes()
        feature_dim = len(self.feature_cols) + len(self.result_cols)
        node_features = torch.zeros((num_nodes, feature_dim))
        
        # Para cada nodo, agregar características de sus aristas salientes
        for node_idx in range(num_nodes):
            # Obtener aristas salientes
            _, out_edges = g.out_edges(node_idx, form='uv')
            
            if len(out_edges) > 0:
                # Obtener características de aristas
                edge_feats = g.edata['features'][out_edges]
                
                # Aplicar agregación
                if aggregation == 'mean' and len(edge_feats) > 0:
                    node_feats = edge_feats.mean(dim=0)
                elif aggregation == 'sum':
                    node_feats = edge_feats.sum(dim=0)
                elif aggregation == 'max' and len(edge_feats) > 0:
                    node_feats, _ = edge_feats.max(dim=0)
                else:
                    # Por defecto, usar media
                    node_feats = edge_feats.mean(dim=0) if len(edge_feats) > 0 else torch.zeros(feature_dim)
                
                node_features[node_idx] = node_feats
        
        return node_features
    
    def update_node_features(self, aggregation: str = 'mean'):
        """
        Actualiza las características de los nodos en el grafo.
        
        Args:
            aggregation: Método de agregación ('mean', 'sum', 'max')
        """
        node_features = self.get_node_features(aggregation)
        self.graph.ndata['h'] = node_features
    
    def get_subgraph_for_teams(self, team_ids: List[str]) -> dgl.DGLGraph:
        """
        Extrae un subgrafo que contiene sólo los equipos especificados.
        
        Args:
            team_ids: Lista de IDs de equipos
            
        Returns:
            Subgrafo DGL
        """
        # Convertir IDs a índices
        node_indices = [self.team_to_idx[team_id] for team_id in team_ids if team_id in self.team_to_idx]
        
        # Extraer subgrafo
        subg = dgl.node_subgraph(self.graph, node_indices)
        
        return subg
    
    def get_team_embedding(self, team_id: str) -> torch.Tensor:
        """
        Obtiene el embedding (características) de un equipo específico.
        
        Args:
            team_id: ID del equipo
            
        Returns:
            Tensor con características del equipo
        """
        if team_id not in self.team_to_idx:
            raise ValueError(f"Equipo no encontrado: {team_id}")
        
        idx = self.team_to_idx[team_id]
        return self.graph.ndata['h'][idx]


class TeamGNN(nn.Module):
    """
    Graph Neural Network para modelar relaciones entre equipos de fútbol.
    
    Este modelo captura las interacciones entre equipos y aprende representaciones
    que reflejan sus estilos de juego y fortalezas relativas.
    """
    
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        gnn_type: str = 'gcn'
    ):
        """
        Inicializa el modelo GNN.
        
        Args:
            in_feats: Dimensión de características de entrada (nodos y aristas)
            hidden_feats: Dimensión de capa oculta
            num_layers: Número de capas GNN
            dropout: Tasa de dropout
            gnn_type: Tipo de GNN ('gcn', 'gat', 'sage')
        """
        super().__init__()
        
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # Capas GNN según el tipo especificado
        self.layers = nn.ModuleList()
        
        if gnn_type == 'gcn':
            # Graph Convolutional Network
            self.layers.append(GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True))
            for _ in range(num_layers - 1):
                self.layers.append(GraphConv(hidden_feats, hidden_feats, allow_zero_in_degree=True))
        
        elif gnn_type == 'gat':
            # Graph Attention Network
            self.layers.append(GATConv(in_feats, hidden_feats // 8, 8, allow_zero_in_degree=True))  # 8 cabezas
            for _ in range(num_layers - 1):
                self.layers.append(GATConv(hidden_feats, hidden_feats // 8, 8, allow_zero_in_degree=True))
        
        elif gnn_type == 'sage':
            # GraphSAGE
            self.layers.append(SAGEConv(in_feats, hidden_feats, 'mean'))
            for _ in range(num_layers - 1):
                self.layers.append(SAGEConv(hidden_feats, hidden_feats, 'mean'))
        
        else:
            raise ValueError(f"Tipo de GNN no soportado: {gnn_type}")
        
        # Capa de predicción de goles
        self.pred_layer = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, 2)  # 2 salidas: goles local, goles visitante
        )
        
        logger.info(f"Modelo TeamGNN inicializado: {gnn_type}, {num_layers} capas, {hidden_feats} features")
    
    def forward(self, g: dgl.DGLGraph, team_indices: torch.Tensor) -> torch.Tensor:
        """
        Pasa el grafo a través del modelo GNN.
        
        Args:
            g: Grafo DGL
            team_indices: Tensor con pares de índices de equipos [batch_size, 2]
            
        Returns:
            Predicción de goles para los partidos entre los equipos especificados
        """
        h = g.ndata['h']
        
        # Pasar a través de capas GNN
        for i, layer in enumerate(self.layers):
            if self.gnn_type == 'gat':
                # Para GAT, necesitamos concatenar las cabezas
                h = layer(g, h).flatten(1)
            else:
                h = layer(g, h)
                
            # Aplicar no linealidad excepto en la última capa
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Actualizar características de nodos en el grafo
        g.ndata['h_gnn'] = h
        
        # Para cada par de equipos, concatenar sus embeddings y predecir goles
        home_embeddings = h[team_indices[:, 0]]
        away_embeddings = h[team_indices[:, 1]]
        
        # Concatenar embeddings
        combined = torch.cat([home_embeddings, away_embeddings], dim=1)
        
        # Predecir goles
        goals_pred = self.pred_layer(combined)
        
        return goals_pred
    
    def get_team_embedding(self, g: dgl.DGLGraph, team_idx: int) -> torch.Tensor:
        """
        Obtiene el embedding aprendido para un equipo específico.
        
        Args:
            g: Grafo DGL
            team_idx: Índice del equipo
            
        Returns:
            Embedding del equipo
        """
        h = g.ndata['h']
        
        # Pasar a través de capas GNN
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if self.gnn_type == 'gat':
                    h = layer(g, h).flatten(1)
                else:
                    h = layer(g, h)
                    
                # Aplicar no linealidad excepto en la última capa
                if i < len(self.layers) - 1:
                    h = F.relu(h)
        
        # Devolver embedding del equipo especificado
        return h[team_idx]


class MatchDataset(Dataset):
    """
    Dataset para entrenamiento con pares de equipos y resultados.
    """
    
    def __init__(
        self,
        team_graph: TeamGraphDataset,
        match_df: pd.DataFrame,
        team_id_home_col: str = 'home_team_id',
        team_id_away_col: str = 'away_team_id',
        result_cols: List[str] = ['home_goals', 'away_goals']
    ):
        """
        Inicializa el dataset de partidos.
        
        Args:
            team_graph: Dataset de grafo de equipos
            match_df: DataFrame con datos de partidos
            team_id_home_col: Nombre de columna con ID del equipo local
            team_id_away_col: Nombre de columna con ID del equipo visitante
            result_cols: Columnas con resultados (goles)
        """
        self.team_graph = team_graph
        self.match_df = match_df
        self.team_id_home_col = team_id_home_col
        self.team_id_away_col = team_id_away_col
        self.result_cols = result_cols
        
        # Filtrar partidos donde ambos equipos están en el grafo
        valid_matches = []
        for _, row in match_df.iterrows():
            home_id = row[team_id_home_col]
            away_id = row[team_id_away_col]
            
            if home_id in team_graph.team_to_idx and away_id in team_graph.team_to_idx:
                valid_matches.append(row)
        
        self.valid_matches = pd.DataFrame(valid_matches)
        
        logger.info(f"MatchDataset inicializado con {len(self.valid_matches)} partidos válidos")
    
    def __len__(self) -> int:
        """Devuelve el número de partidos."""
        return len(self.valid_matches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve un par de equipos y el resultado correspondiente.
        
        Args:
            idx: Índice del partido
            
        Returns:
            Tupla con (índices de equipos, goles)
        """
        match = self.valid_matches.iloc[idx]
        
        # Obtener índices de equipos
        home_idx = self.team_graph.team_to_idx[match[self.team_id_home_col]]
        away_idx = self.team_graph.team_to_idx[match[self.team_id_away_col]]
        team_indices = torch.tensor([home_idx, away_idx])
        
        # Obtener goles
        goals = torch.tensor([float(match[col]) for col in self.result_cols], dtype=torch.float)
        
        return team_indices, goals


class TeamGNNTrainer:
    """
    Clase para entrenar y evaluar modelos GNN para equipos.
    """
    
    def __init__(
        self,
        model: TeamGNN,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Inicializa el entrenador.
        
        Args:
            model: Modelo TeamGNN a entrenar
            learning_rate: Tasa de aprendizaje
            weight_decay: Regularización L2
            device: Dispositivo de entrenamiento
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()  # Para predicción de goles
        
        logger.info(f"Entrenador inicializado en dispositivo: {device}")
    
    def train_epoch(self, graph: dgl.DGLGraph, dataloader: DataLoader) -> float:
        """
        Entrena el modelo durante una época.
        
        Args:
            graph: Grafo DGL
            dataloader: DataLoader con datos de entrenamiento
            
        Returns:
            Pérdida media de la época
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        graph = graph.to(self.device)
        
        for team_indices, goals in dataloader:
            team_indices = team_indices.to(self.device)
            goals = goals.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(graph, team_indices)
            
            # Calcular pérdida
            loss = self.criterion(predictions, goals)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def evaluate(self, graph: dgl.DGLGraph, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evalúa el modelo en un conjunto de datos.
        
        Args:
            graph: Grafo DGL
            dataloader: DataLoader con datos de evaluación
            
        Returns:
            Diccionario con métricas de evaluación
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_goals = []
        
        graph = graph.to(self.device)
        
        with torch.no_grad():
            for team_indices, goals in dataloader:
                team_indices = team_indices.to(self.device)
                goals = goals.to(self.device)
                
                # Forward pass
                predictions = self.model(graph, team_indices)
                
                # Calcular pérdida
                loss = self.criterion(predictions, goals)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Guardar predicciones y valores reales
                all_predictions.append(predictions.cpu())
                all_goals.append(goals.cpu())
        
        # Concatenar resultados
        all_predictions = torch.cat(all_predictions, dim=0)
        all_goals = torch.cat(all_goals, dim=0)
        
        # Calcular métricas
        mse = F.mse_loss(all_predictions, all_goals).item()
        mae = F.l1_loss(all_predictions, all_goals).item()
        
        # MSE separado para goles local y visitante
        home_mse = F.mse_loss(all_predictions[:, 0], all_goals[:, 0]).item()
        away_mse = F.mse_loss(all_predictions[:, 1], all_goals[:, 1]).item()
        
        metrics = {
            'loss': total_loss / max(1, num_batches),
            'mse': mse,
            'mae': mae,
            'home_mse': home_mse,
            'away_mse': away_mse
        }
        
        return metrics
    
    def train(
        self,
        graph: dgl.DGLGraph,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
        patience: int = 5,
        model_save_path: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Entrena el modelo durante un número de épocas.
        
        Args:
            graph: Grafo DGL
            train_loader: DataLoader con datos de entrenamiento
            val_loader: DataLoader con datos de validación
            num_epochs: Número máximo de épocas
            patience: Épocas sin mejora antes de parar
            model_save_path: Ruta para guardar el mejor modelo
            
        Returns:
            Diccionario con historial de entrenamiento
        """
        train_losses = []
        val_metrics_history = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Entrenar una época
            train_loss = self.train_epoch(graph, train_loader)
            train_losses.append(train_loss)
            
            # Evaluar
            val_metrics = self.evaluate(graph, val_loader)
            val_metrics_history.append(val_metrics)
            
            val_loss = val_metrics['loss']
            
            # Mostrar progreso
            logger.info(f"Época {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val MSE: {val_metrics['mse']:.4f}, "
                       f"Home MSE: {val_metrics['home_mse']:.4f}, "
                       f"Away MSE: {val_metrics['away_mse']:.4f}")
            
            # Verificar mejora
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
            'train_losses': train_losses,
            'val_metrics': val_metrics_history
        }


class TeamGNNPredictor:
    """
    Clase para realizar predicciones con modelos GNN pre-entrenados.
    """
    
    def __init__(
        self,
        model_path: str,
        team_graph: TeamGraphDataset,
        in_features: int,
        hidden_features: int = 64,
        num_layers: int = 2,
        gnn_type: str = 'gcn',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Inicializa el predictor con un modelo pre-entrenado.
        
        Args:
            model_path: Ruta al modelo guardado
            team_graph: Dataset con grafo de equipos
            in_features: Dimensión de características de entrada
            hidden_features: Dimensión de capa oculta
            num_layers: Número de capas GNN
            gnn_type: Tipo de GNN ('gcn', 'gat', 'sage')
            device: Dispositivo para inferencia
        """
        self.device = device
        self.team_graph = team_graph
        
        # Inicializar modelo
        self.model = TeamGNN(
            in_feats=in_features,
            hidden_feats=hidden_features,
            num_layers=num_layers,
            gnn_type=gnn_type
        ).to(device)
        
        # Cargar pesos pre-entrenados
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        logger.info(f"Predictor TeamGNN cargado desde {model_path}")
    
    def predict_match(self, home_team_id: str, away_team_id: str) -> Tuple[float, float]:
        """
        Predice el resultado de un partido entre dos equipos.
        
        Args:
            home_team_id: ID del equipo local
            away_team_id: ID del equipo visitante
            
        Returns:
            Tupla con (goles local, goles visitante)
        """
        # Verificar que los equipos existen en el grafo
        if home_team_id not in self.team_graph.team_to_idx:
            raise ValueError(f"Equipo local no encontrado: {home_team_id}")
        if away_team_id not in self.team_graph.team_to_idx:
            raise ValueError(f"Equipo visitante no encontrado: {away_team_id}")
        
        # Obtener índices
        home_idx = self.team_graph.team_to_idx[home_team_id]
        away_idx = self.team_graph.team_to_idx[away_team_id]
        
        # Crear tensor de índices
        team_indices = torch.tensor([[home_idx, away_idx]], dtype=torch.long).to(self.device)
        
        # Realizar predicción
        graph = self.team_graph.graph.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(graph, team_indices)
        
        # Extraer goles predichos
        home_goals, away_goals = prediction[0].cpu().numpy()
        
        # Asegurar valores no negativos
        home_goals = max(0, home_goals)
        away_goals = max(0, away_goals)
        
        return float(home_goals), float(away_goals)
    
    def get_team_style(self, team_id: str) -> Dict[str, float]:
        """
        Obtiene una representación del estilo de juego de un equipo.
        
        Args:
            team_id: ID del equipo
            
        Returns:
            Diccionario con características de estilo de juego
        """
        # Verificar que el equipo existe
        if team_id not in self.team_graph.team_to_idx:
            raise ValueError(f"Equipo no encontrado: {team_id}")
        
        # Obtener embedding del equipo
        team_idx = self.team_graph.team_to_idx[team_id]
        graph = self.team_graph.graph.to(self.device)
        
        embedding = self.model.get_team_embedding(graph, team_idx).cpu().numpy()
        
        # Normalizar embedding para interpretabilidad
        embedding_norm = embedding / np.linalg.norm(embedding)
        
        # Calcular características estilísticas a partir del embedding
        # Esto es una simplificación, en la práctica se necesitaría un análisis más sofisticado
        style = {
            'attacking_strength': float(np.mean(embedding_norm[:embedding_norm.shape[0]//3])),
            'defensive_solidity': float(np.mean(embedding_norm[embedding_norm.shape[0]//3:2*embedding_norm.shape[0]//3])),
            'tactical_flexibility': float(np.std(embedding_norm)),
            'home_advantage': float(np.mean(embedding_norm[2*embedding_norm.shape[0]//3:])),
            'consistency': float(1.0 - np.std(embedding_norm))
        }
        
        return style
    
    def get_matchup_analysis(self, home_team_id: str, away_team_id: str) -> Dict[str, Any]:
        """
        Analiza un enfrentamiento entre dos equipos.
        
        Args:
            home_team_id: ID del equipo local
            away_team_id: ID del equipo visitante
            
        Returns:
            Diccionario con análisis del enfrentamiento
        """
        # Predecir resultado
        home_goals, away_goals = self.predict_match(home_team_id, away_team_id)
        
        # Obtener estilos de juego
        home_style = self.get_team_style(home_team_id)
        away_style = self.get_team_style(away_team_id)
        
        # Calcular ventajas comparativas
        comparative = {
            'attacking': home_style['attacking_strength'] - away_style['defensive_solidity'],
            'defensive': home_style['defensive_solidity'] - away_style['attacking_strength'],
            'tactical': home_style['tactical_flexibility'] - away_style['tactical_flexibility']
        }
        
        # Generar análisis
        analysis = {
            'predicted_score': {
                'home': home_goals,
                'away': away_goals
            },
            'team_styles': {
                'home': home_style,
                'away': away_style
            },
            'comparative_advantage': comparative,
            'key_factors': self._identify_key_factors(home_style, away_style)
        }
        
        return analysis
    
    def _identify_key_factors(self, home_style: Dict[str, float], away_style: Dict[str, float]) -> List[str]:
        """
        Identifica factores clave en un enfrentamiento.
        
        Args:
            home_style: Estilo de juego del equipo local
            away_style: Estilo de juego del equipo visitante
            
        Returns:
            Lista de factores clave
        """
        factors = []
        
        # Comparar características y encontrar diferencias significativas
        if home_style['attacking_strength'] - away_style['defensive_solidity'] > 0.2:
            factors.append("Ventaja ofensiva del equipo local")
        
        if away_style['attacking_strength'] - home_style['defensive_solidity'] > 0.2:
            factors.append("Ventaja ofensiva del equipo visitante")
        
        if home_style['home_advantage'] > 0.6:
            factors.append("Fuerte ventaja local")
        
        if abs(home_style['tactical_flexibility'] - away_style['tactical_flexibility']) > 0.3:
            more_flexible = "local" if home_style['tactical_flexibility'] > away_style['tactical_flexibility'] else "visitante"
            factors.append(f"Mayor flexibilidad táctica del equipo {more_flexible}")
        
        if abs(home_style['consistency'] - away_style['consistency']) > 0.3:
            more_consistent = "local" if home_style['consistency'] > away_style['consistency'] else "visitante"
            factors.append(f"Mayor consistencia del equipo {more_consistent}")
        
        return factors


def prepare_data_for_gnn(
    match_df: pd.DataFrame,
    team_id_home_col: str = 'home_team_id',
    team_id_away_col: str = 'away_team_id',
    result_cols: List[str] = ['home_goals', 'away_goals'],
    feature_cols: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    time_window: Optional[int] = None
) -> Dict[str, Any]:
    """
    Prepara datos para entrenamiento y evaluación de GNN.
    
    Args:
        match_df: DataFrame con datos de partidos
        team_id_home_col: Nombre de columna con ID del equipo local
        team_id_away_col: Nombre de columna con ID del equipo visitante
        result_cols: Columnas con resultados (goles)
        feature_cols: Columnas adicionales para features
        train_ratio: Proporción de datos para entrenamiento
        val_ratio: Proporción de datos para validación
        batch_size: Tamaño de batch
        time_window: Ventana de tiempo para filtrar partidos (en días)
        
    Returns:
        Diccionario con datos preparados
    """
    # Crear grafo de equipos
    team_graph = TeamGraphDataset(
        match_df=match_df,
        team_id_home_col=team_id_home_col,
        team_id_away_col=team_id_away_col,
        result_cols=result_cols,
        feature_cols=feature_cols,
        time_window=time_window
    )
    
    # Inicializar características de nodos
    team_graph.update_node_features(aggregation='mean')
    
    # Crear dataset de partidos
    match_dataset = MatchDataset(
        team_graph=team_graph,
        match_df=match_df,
        team_id_home_col=team_id_home_col,
        team_id_away_col=team_id_away_col,
        result_cols=result_cols
    )
    
    # Dividir en conjuntos de entrenamiento, validación y test
    num_samples = len(match_dataset)
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        match_dataset, [train_size, val_size, test_size]
    )
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'team_graph': team_graph,
        'graph': team_graph.graph,
        'dataloaders': {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        },
        'feature_dim': len(result_cols) + (len(feature_cols) if feature_cols else 0)
    }


def train_team_gnn(
    match_df: pd.DataFrame,
    model_save_path: str,
    hidden_feats: int = 64,
    num_layers: int = 2,
    gnn_type: str = 'gcn',
    learning_rate: float = 0.001,
    batch_size: int = 32,
    num_epochs: int = 30,
    patience: int = 5
) -> Dict[str, Any]:
    """
    Entrena un modelo GNN para predecir resultados de partidos.
    
    Args:
        match_df: DataFrame con datos de partidos
        model_save_path: Ruta para guardar el modelo
        hidden_feats: Dimensión de capa oculta
        num_layers: Número de capas GNN
        gnn_type: Tipo de GNN ('gcn', 'gat', 'sage')
        learning_rate: Tasa de aprendizaje
        batch_size: Tamaño de batch
        num_epochs: Número de épocas máximo
        patience: Épocas sin mejora antes de parar
        
    Returns:
        Diccionario con resultados del entrenamiento
    """
    # Preparar datos
    logger.info("Preparando datos para GNN...")
    data = prepare_data_for_gnn(
        match_df=match_df,
        batch_size=batch_size
    )
    
    team_graph = data['team_graph']
    graph = data['graph']
    train_loader = data['dataloaders']['train']
    val_loader = data['dataloaders']['val']
    test_loader = data['dataloaders']['test']
    in_feats = graph.ndata['h'].shape[1]
    
    # Inicializar modelo
    logger.info(f"Inicializando modelo TeamGNN ({gnn_type})...")
    model = TeamGNN(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        num_layers=num_layers,
        gnn_type=gnn_type
    )
    
    # Inicializar entrenador
    trainer = TeamGNNTrainer(
        model=model,
        learning_rate=learning_rate
    )
    
    # Entrenar modelo
    logger.info("Iniciando entrenamiento...")
    training_history = trainer.train(
        graph=graph,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        patience=patience,
        model_save_path=model_save_path
    )
    
    # Evaluar en conjunto de test
    logger.info("Evaluando modelo en conjunto de test...")
    test_metrics = trainer.evaluate(graph, test_loader)
    
    # Crear directorios si no existen
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Guardar metadatos del modelo
    metadata = {
        'model_params': {
            'in_features': in_feats,
            'hidden_features': hidden_feats,
            'num_layers': num_layers,
            'gnn_type': gnn_type
        },
        'test_metrics': test_metrics,
        'num_teams': len(team_graph.teams),
        'team_mapping': team_graph.team_to_idx,
        'created_date': datetime.now().isoformat()
    }
    
    metadata_path = os.path.splitext(model_save_path)[0] + '_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'training_history': training_history,
        'test_metrics': test_metrics,
        'model_path': model_save_path,
        'metadata_path': metadata_path
    }


def predict_with_team_gnn(
    model_path: str,
    match_df: pd.DataFrame,
    home_team_id: str,
    away_team_id: str
) -> Dict[str, Any]:
    """
    Realiza predicción para un partido utilizando modelo GNN pre-entrenado.
    
    Args:
        model_path: Ruta al modelo guardado
        match_df: DataFrame con datos de partidos para construir grafo
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        
    Returns:
        Diccionario con predicción y análisis
    """
    # Cargar metadatos del modelo
    metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No se encontraron metadatos del modelo en {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Crear grafo de equipos
    team_graph = TeamGraphDataset(
        match_df=match_df
    )
    
    # Inicializar predictor
    predictor = TeamGNNPredictor(
        model_path=model_path,
        team_graph=team_graph,
        in_features=metadata['model_params']['in_features'],
        hidden_features=metadata['model_params']['hidden_features'],
        num_layers=metadata['model_params']['num_layers'],
        gnn_type=metadata['model_params']['gnn_type']
    )
    
    # Realizar predicción y análisis
    analysis = predictor.get_matchup_analysis(home_team_id, away_team_id)
    
    return analysis


def integrate_gnn_with_transformer(
    gnn_prediction: Dict[str, Any],
    transformer_prediction: Dict[str, Any],
    ensemble_prediction: Dict[str, Any],
    gnn_weight: float = 0.2,
    transformer_weight: float = 0.3,
    ensemble_weight: float = 0.5
) -> Dict[str, Any]:
    """
    Integra predicciones de GNN, Transformer y Ensemble.
    
    Args:
        gnn_prediction: Predicción del modelo GNN
        transformer_prediction: Predicción del modelo Transformer
        ensemble_prediction: Predicción del ensemble
        gnn_weight: Peso para predicción GNN
        transformer_weight: Peso para predicción Transformer
        ensemble_weight: Peso para predicción Ensemble
        
    Returns:
        Predicción integrada
    """
    # Verificar que las predicciones incluyen los campos necesarios
    if 'predicted_score' not in gnn_prediction:
        raise ValueError("La predicción GNN debe incluir 'predicted_score'")
    
    if 'predicted_home_goals' not in transformer_prediction or 'predicted_away_goals' not in transformer_prediction:
        raise ValueError("La predicción Transformer debe incluir 'predicted_home_goals' y 'predicted_away_goals'")
    
    if 'predicted_home_goals' not in ensemble_prediction or 'predicted_away_goals' not in ensemble_prediction:
        raise ValueError("La predicción Ensemble debe incluir 'predicted_home_goals' y 'predicted_away_goals'")
    
    # Normalizar pesos
    total_weight = gnn_weight + transformer_weight + ensemble_weight
    gnn_weight /= total_weight
    transformer_weight /= total_weight
    ensemble_weight /= total_weight
    
    # Obtener valores de predicciones
    gnn_home = gnn_prediction['predicted_score']['home']
    gnn_away = gnn_prediction['predicted_score']['away']
    
    transformer_home = transformer_prediction['predicted_home_goals']
    transformer_away = transformer_prediction['predicted_away_goals']
    
    ensemble_home = ensemble_prediction['predicted_home_goals']
    ensemble_away = ensemble_prediction['predicted_away_goals']
    
    # Calcular predicción ponderada
    integrated_home = (gnn_home * gnn_weight) + (transformer_home * transformer_weight) + (ensemble_home * ensemble_weight)
    integrated_away = (gnn_away * gnn_weight) + (transformer_away * transformer_weight) + (ensemble_away * ensemble_weight)
    
    # Crear resultado integrado
    integrated = {
        'predicted_home_goals': integrated_home,
        'predicted_away_goals': integrated_away,
        'integration_weights': {
            'gnn': gnn_weight,
            'transformer': transformer_weight,
            'ensemble': ensemble_weight
        },
        'component_predictions': {
            'gnn': {'home': gnn_home, 'away': gnn_away},
            'transformer': {'home': transformer_home, 'away': transformer_away},
            'ensemble': {'home': ensemble_home, 'away': ensemble_away}
        },
        'team_style_analysis': gnn_prediction.get('team_styles', {}),
        'key_factors': gnn_prediction.get('key_factors', [])
    }
    
    return integrated
