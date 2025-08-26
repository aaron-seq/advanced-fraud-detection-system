#!/usr/bin/env python3
"""
Graph Neural Network for Credit Card Fraud Detection
Leveraging transaction networks and user relationships for advanced fraud detection
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import logging
import json
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionGraphBuilder:
    """Build transaction graphs from credit card data"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.node_features = {}
        self.edge_features = {}
        self.graph = nx.Graph()
    
    def create_transaction_graph(self, df: pd.DataFrame, method: str = 'user_based') -> Data:
        """
        Create graph from transaction data
        
        Methods:
        - 'user_based': Nodes are users, edges represent shared transaction patterns
        - 'transaction_based': Nodes are transactions, edges represent similarities
        - 'hybrid': Combines both approaches
        """
        
        if method == 'user_based':
            return self._create_user_based_graph(df)
        elif method == 'transaction_based':
            return self._create_transaction_based_graph(df)
        elif method == 'hybrid':
            return self._create_hybrid_graph(df)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _create_user_based_graph(self, df: pd.DataFrame) -> Data:
        """Create graph where nodes represent users/cards"""
        logger.info("Creating user-based transaction graph...")
        
        # For simplicity, we'll create synthetic user IDs
        # In real data, you'd have actual user/card identifiers
        n_users = min(1000, len(df) // 10)  # Assume each user has ~10 transactions
        user_ids = np.random.randint(0, n_users, len(df))
        df = df.copy()
        df['user_id'] = user_ids
        
        # Aggregate features by user
        user_features = df.groupby('user_id').agg({
            'Time': ['mean', 'std', 'min', 'max'],
            'Amount': ['mean', 'std', 'sum', 'count'],
            'Class': 'max',  # If any transaction is fraud, user is suspicious
            **{f'V{i}': ['mean', 'std'] for i in range(1, 29)}
        }).fillna(0)
        
        # Flatten column names
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
        
        # Create node features matrix
        node_features = torch.FloatTensor(user_features.drop('Class_max', axis=1).values)
        node_labels = torch.LongTensor(user_features['Class_max'].values)
        
        # Create edges based on transaction pattern similarity
        edges = []
        edge_weights = []
        
        logger.info("Computing user similarities...")
        feature_matrix = user_features.drop('Class_max', axis=1).values
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Use a subset for efficiency
        n_subset = min(500, len(feature_matrix_scaled))
        indices = np.random.choice(len(feature_matrix_scaled), n_subset, replace=False)
        subset_features = feature_matrix_scaled[indices]
        
        for i in range(len(subset_features)):
            for j in range(i+1, len(subset_features)):
                # Calculate cosine similarity
                similarity = np.dot(subset_features[i], subset_features[j]) / (
                    np.linalg.norm(subset_features[i]) * np.linalg.norm(subset_features[j]) + 1e-8
                )
                
                if similarity > self.similarity_threshold:
                    edges.extend([[indices[i], indices[j]], [indices[j], indices[i]]])
                    edge_weights.extend([similarity, similarity])
        
        # Convert to tensor format
        if edges:
            edge_index = torch.LongTensor(edges).t().contiguous()
            edge_attr = torch.FloatTensor(edge_weights)
        else:
            # Create some random edges if no similarities found
            n_edges = min(100, n_subset * 2)
            source_nodes = np.random.choice(n_subset, n_edges)
            target_nodes = np.random.choice(n_subset, n_edges)
            edge_index = torch.LongTensor([source_nodes, target_nodes])
            edge_attr = torch.FloatTensor(np.random.uniform(0.5, 1.0, n_edges))
        
        # Create PyTorch Geometric data object
        data = Data(
            x=node_features[:n_subset],
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_labels[:n_subset]
        )
        
        logger.info(f"Created user-based graph: {data.num_nodes} nodes, {data.num_edges} edges")
        return data
    
    def _create_transaction_based_graph(self, df: pd.DataFrame) -> Data:
        """Create graph where nodes represent individual transactions"""
        logger.info("Creating transaction-based graph...")
        
        # Limit size for computational efficiency
        n_transactions = min(2000, len(df))
        df_subset = df.sample(n_transactions, random_state=42).reset_index(drop=True)
        
        # Use transaction features as node features
        feature_cols = [col for col in df_subset.columns if col.startswith('V')] + ['Time', 'Amount']
        node_features = torch.FloatTensor(df_subset[feature_cols].values)
        node_labels = torch.LongTensor(df_subset['Class'].values)
        
        # Create edges based on transaction similarity
        edges = []
        edge_weights = []
        
        # Scale features for similarity computation
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_subset[feature_cols].values)
        
        logger.info("Computing transaction similarities...")
        # Use k-nearest neighbors for efficiency
        from sklearn.neighbors import NearestNeighbors
        
        k = min(10, n_transactions - 1)  # Number of nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(features_scaled)
        distances, indices = nbrs.kneighbors(features_scaled)
        
        for i in range(n_transactions):
            for j in range(1, k):  # Skip self (j=0)
                neighbor_idx = indices[i, j]
                distance = distances[i, j]
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity > self.similarity_threshold:
                    edges.extend([[i, neighbor_idx], [neighbor_idx, i]])
                    edge_weights.extend([similarity, similarity])
        
        # Convert to tensor format
        if edges:
            edge_index = torch.LongTensor(edges).t().contiguous()
            edge_attr = torch.FloatTensor(edge_weights)
        else:
            # Create some random edges if no similarities found
            n_edges = n_transactions * 2
            source_nodes = np.random.choice(n_transactions, n_edges)
            target_nodes = np.random.choice(n_transactions, n_edges)
            edge_index = torch.LongTensor([source_nodes, target_nodes])
            edge_attr = torch.FloatTensor(np.random.uniform(0.5, 1.0, n_edges))
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_labels
        )
        
        logger.info(f"Created transaction-based graph: {data.num_nodes} nodes, {data.num_edges} edges")
        return data
    
    def _create_hybrid_graph(self, df: pd.DataFrame) -> Data:
        """Create hybrid graph combining users and transactions"""
        logger.info("Creating hybrid transaction graph...")
        # For simplicity, return transaction-based graph
        # In practice, you'd create a heterogeneous graph with both user and transaction nodes
        return self._create_transaction_based_graph(df)

class GraphFraudDetector(nn.Module):
    """Graph Neural Network for fraud detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2, 
                 model_type: str = 'GCN', num_layers: int = 3, dropout: float = 0.3):
        super(GraphFraudDetector, self).__init__()
        
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        
        if model_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        elif model_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        elif model_type == 'SAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1:  # No dropout after last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=-1)

class GNNFraudDetectionSystem:
    """Complete GNN-based fraud detection system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.model = None
        self.graph_builder = TransactionGraphBuilder(
            similarity_threshold=self.config['similarity_threshold']
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def _default_config(self):
        return {
            'model_type': 'GAT',  # GCN, GAT, or SAGE
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 5e-4,
            'epochs': 200,
            'batch_size': 32,
            'patience': 20,
            'similarity_threshold': 0.7,
            'graph_method': 'transaction_based'
        }
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[Data, Data]:
        """Prepare graph data for training and testing"""
        logger.info("Preparing graph data...")
        
        # Create graph
        graph_data = self.graph_builder.create_transaction_graph(
            df, method=self.config['graph_method']
        )
        
        # Split nodes for train/test
        num_nodes = graph_data.num_nodes
        indices = torch.randperm(num_nodes)
        
        train_size = int(0.8 * num_nodes)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # Create train and test masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        
        graph_data.train_mask = train_mask
        graph_data.test_mask = test_mask
        
        return graph_data, graph_data  # Same graph, different masks
    
    def create_model(self, input_dim: int):
        """Create GNN model"""
        model = GraphFraudDetector(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_classes=2,
            model_type=self.config['model_type'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        return model
    
    def train_model(self, train_data: Data) -> Dict:
        """Train the GNN model"""
        logger.info("Training GNN model...")
        
        # Create model
        self.model = self.create_model(train_data.num_node_features)
        
        # Setup training
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        criterion = nn.NLLLoss()
        
        # Move data to device
        train_data = train_data.to(self.device)
        
        # Training loop
        self.model.train()
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            optimizer.zero_grad()
            
            # Forward pass
            out = self.model(train_data)
            
            # Calculate loss (only on training nodes)
            train_loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            
            # Evaluate
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(train_data)
                
                # Training accuracy
                train_pred = out[train_data.train_mask].max(1)[1]
                train_acc = train_pred.eq(train_data.y[train_data.train_mask]).float().mean()
                
                # Validation accuracy (using test mask as validation)
                val_pred = val_out[train_data.test_mask].max(1)[1]
                val_acc = val_pred.eq(train_data.y[train_data.test_mask]).float().mean()
                
                # Validation loss
                val_loss = criterion(val_out[train_data.test_mask], train_data.y[train_data.test_mask])
            
            # Store metrics
            training_history['train_loss'].append(train_loss.item())
            training_history['train_acc'].append(train_acc.item())
            training_history['val_loss'].append(val_loss.item())
            training_history['val_acc'].append(val_acc.item())
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_gnn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % 20 == 0:
                logger.info(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, '
                          f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, '
                          f'Val Acc: {val_acc:.4f}')
            
            self.model.train()
        
        # Load best model
        self.model.load_state_dict(torch.load('best_gnn_model.pth'))
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        
        return training_history
    
    def evaluate_model(self, test_data: Data) -> Dict:
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        logger.info("Evaluating GNN model...")
        
        self.model.eval()
        test_data = test_data.to(self.device)
        
        with torch.no_grad():
            # Get predictions
            out = self.model(test_data)
            pred_probs = torch.exp(out)  # Convert log softmax to probabilities
            pred_labels = out.max(1)[1]
            
            # Extract test results
            test_pred_probs = pred_probs[test_data.test_mask]
            test_pred_labels = pred_labels[test_data.test_mask]
            test_true_labels = test_data.y[test_data.test_mask]
            
            # Move to CPU for sklearn metrics
            y_true = test_true_labels.cpu().numpy()
            y_pred = test_pred_labels.cpu().numpy()
            y_prob = test_pred_probs[:, 1].cpu().numpy()  # Probability of fraud class
            
            # Calculate metrics
            accuracy = (y_pred == y_true).mean()
            auc = roc_auc_score(y_true, y_prob)
            
            # Classification report
            class_report = classification_report(y_true, y_pred, output_dict=True)
            
            results = {
                'accuracy': accuracy,
                'auc': auc,
                'classification_report': class_report,
                'predictions': y_pred,
                'probabilities': y_prob,
                'true_labels': y_true
            }
            
            logger.info(f"Test Accuracy: {accuracy:.4f}, Test AUC: {auc:.4f}")
            
            return results

def generate_synthetic_credit_card_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic credit card transaction data for testing"""
    logger.info(f"Generating {n_samples} synthetic transactions...")
    
    # Generate features
    data = {}
    
    # Time feature (seconds)
    data['Time'] = np.random.exponential(1000, n_samples)
    
    # Amount feature
    data['Amount'] = np.random.exponential(50, n_samples)
    
    # V1-V28 features (PCA components)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create fraud labels (1% fraud rate)
    fraud_indices = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)
    df['Class'] = 0
    df.loc[fraud_indices, 'Class'] = 1
    
    # Make fraudulent transactions more distinct
    for col in df.columns:
        if col.startswith('V') or col == 'Amount':
            df.loc[fraud_indices, col] *= np.random.uniform(2, 5, len(fraud_indices))
    
    logger.info(f"Created dataset with {df['Class'].sum()} fraud cases ({df['Class'].mean()*100:.2f}%)")
    
    return df

def main():
    """Main function to run GNN fraud detection"""
    print("Graph Neural Network Credit Card Fraud Detection")
    print("=" * 60)
    
    # Configuration
    config = {
        'model_type': 'GAT',  # Try GCN, GAT, or SAGE
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 100,  # Reduced for demo
        'patience': 15,
        'similarity_threshold': 0.7,
        'graph_method': 'transaction_based'
    }
    
    # Generate synthetic data (or load real data)
    df = generate_synthetic_credit_card_data(n_samples=5000)  # Smaller for demo
    
    # Initialize GNN system
    gnn_system = GNNFraudDetectionSystem(config)
    
    try:
        # Prepare data
        train_data, test_data = gnn_system.prepare_data(df)
        
        # Train model
        training_history = gnn_system.train_model(train_data)
        
        # Evaluate model
        results = gnn_system.evaluate_model(test_data)
        
        # Print results
        print(f"\nGNN Model Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"AUC: {results['auc']:.4f}")
        
        if results['classification_report']:
            print(f"\nDetailed Classification Report:")
            print(f"Precision (Fraud): {results['classification_report']['1']['precision']:.4f}")
            print(f"Recall (Fraud): {results['classification_report']['1']['recall']:.4f}")
            print(f"F1-Score (Fraud): {results['classification_report']['1']['f1-score']:.4f}")
        
        # Save model and results
        torch.save(gnn_system.model.state_dict(), 'gnn_fraud_model.pth')
        
        with open('gnn_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {
                'accuracy': float(results['accuracy']),
                'auc': float(results['auc']),
                'config': config,
                'classification_report': results['classification_report']
            }
            json.dump(serializable_results, f, indent=2)
        
        print("\nModel and results saved successfully!")
        
    except Exception as e:
        logger.error(f"Error in GNN fraud detection: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
