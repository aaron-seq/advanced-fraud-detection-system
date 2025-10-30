# src/graph_neural_network_fraud.py

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TransactionGraphBuilder:
    """
    Builds a transaction graph from credit card data for use in a GNN.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def create_graph(self, df: pd.DataFrame, method: str = 'transaction_based') -> Data:
        """
        Creates a graph from transaction data using the specified method.

        Args:
            df (pd.DataFrame): The transaction data.
            method (str): The graph construction method ('transaction_based' or 'user_based').

        Returns:
            Data: A PyTorch Geometric Data object representing the graph.
        """
        if method == 'transaction_based':
            return self._create_transaction_based_graph(df)
        elif method == 'user_based':
            return self._create_user_based_graph(df)
        else:
            raise ValueError(f"Unknown graph construction method: {method}")

    def _create_transaction_based_graph(self, df: pd.DataFrame) -> Data:
        """Creates a graph where nodes represent individual transactions."""
        df_subset = df.sample(min(2000, len(df)), random_state=42)

        feature_cols = [col for col in df_subset.columns if col.startswith('V')] + ['Time', 'Amount']
        node_features = torch.FloatTensor(df_subset[feature_cols].values)
        node_labels = torch.LongTensor(df_subset['Class'].values)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_subset[feature_cols])

        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=10, metric='cosine').fit(features_scaled)
        distances, indices = nbrs.kneighbors(features_scaled)

        edges = []
        for i in range(len(features_scaled)):
            for j in range(1, 10):
                if 1 - distances[i, j] > self.similarity_threshold:
                    edges.append([i, indices[i, j]])

        edge_index = torch.LongTensor(edges).t().contiguous()
        return Data(x=node_features, edge_index=edge_index, y=node_labels)

    def _create_user_based_graph(self, df: pd.DataFrame) -> Data:
        """Creates a graph where nodes represent users, not implemented in detail."""
        logging.warning("User-based graph construction is not fully implemented and will fall back to transaction-based.")
        return self._create_transaction_based_graph(df)

class GraphFraudDetector(nn.Module):
    """
    A Graph Neural Network for fraud detection, supporting GCN, GAT, and SAGE convolutions.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, model_type: str = 'GCN', num_layers: int = 3, dropout: float = 0.3):
        super(GraphFraudDetector, self).__init__()

        self.convs = nn.ModuleList()
        conv_layer = {'GCN': GCNConv, 'GAT': GATConv, 'SAGE': SAGEConv}[model_type]

        self.convs.append(conv_layer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(conv_layer(hidden_dim, hidden_dim))

        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class GNNSystem:
    """
    Manages the training and evaluation of the GNN-based fraud detection system.
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _default_config(self) -> Dict:
        """Provides a default configuration for the GNN system."""
        return {
            'model_type': 'GAT',
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 15
        }

    def train(self, df: pd.DataFrame) -> Dict:
        """Trains the GNN model on the provided transaction data."""
        builder = TransactionGraphBuilder()
        graph_data = builder.create_graph(df)

        self.model = GraphFraudDetector(
            input_dim=graph_data.num_node_features,
            hidden_dim=self.config['hidden_dim'],
            model_type=self.config['model_type'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        graph_data.to(self.device)
        self.model.train()

        for epoch in range(self.config['epochs']):
            optimizer.zero_grad()
            out = self.model(graph_data)
            loss = F.nll_loss(out, graph_data.y)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logging.info(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

        return self.evaluate(graph_data)

    def evaluate(self, data: Data) -> Dict:
        """Evaluates the trained GNN model."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            y_true = data.y.cpu().numpy()
            y_prob = torch.exp(out)[:, 1].cpu().numpy()

            auc = roc_auc_score(y_true, y_prob)
            report = classification_report(y_true, (y_prob > 0.5).astype(int), output_dict=True)

            return {'auc': auc, 'classification_report': report}

def main():
    """Main function to run GNN-based fraud detection."""
    df = pd.read_csv('creditcard.csv')

    config = {'model_type': 'GAT', 'epochs': 100}
    gnn_system = GNNSystem(config)

    results = gnn_system.train(df)

    logging.info(f"Final GNN Model AUC: {results['auc']:.4f}")
    logging.info(f"Classification Report:\n{pd.DataFrame(results['classification_report']).transpose()}")

if __name__ == "__main__":
    main()
