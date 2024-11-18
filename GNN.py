import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm

class APKDataset:
    def __init__(self, json_path):
        self.graphs = self.load_json(json_path)
        self.feature_dim = self.get_feature_dimension()
        print(f"Feature dimension: {self.feature_dim}")
        
    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def get_feature_dimension(self):
        """Determine the feature dimension from the first valid node features"""
        for graph in self.graphs.values():
            for node in graph['nodes']:
                if node['features'] and len(node['features']) > 0:
                    return len(node['features'])
        raise ValueError("No valid features found in any node")
    
    def pad_or_truncate_features(self, features):
        """Pad with zeros or truncate features to match feature_dim"""
        if not features:
            return np.zeros(self.feature_dim)
        features = np.array(features)
        if len(features) > self.feature_dim:
            return features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            return np.pad(features, (0, self.feature_dim - len(features)))
        return features

    def preprocess_graph(self, graph):
        """Preprocess a single graph to ensure valid features and edges"""
        # Create node mapping
        node_mapping = {node['id']: idx for idx, node in enumerate(graph['nodes'])}
        
        # Process nodes
        processed_features = []
        valid_nodes = []
        new_mapping = {}
        
        for idx, node in enumerate(graph['nodes']):
            features = self.pad_or_truncate_features(node['features'])
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                continue
            processed_features.append(features)
            valid_nodes.append(node['id'])
            new_mapping[node['id']] = len(new_mapping)
        
        if not processed_features:
            # If no valid nodes, create a dummy node with zero features
            processed_features = [np.zeros(self.feature_dim)]
            new_mapping = {'dummy': 0}
            valid_nodes = ['dummy']
        
        # Process edges
        valid_edges = []
        for edge in graph['edges']:
            source = edge['source']
            target = edge['target']
            if source in new_mapping and target in new_mapping:
                valid_edges.append([new_mapping[source], new_mapping[target]])
        
        if not valid_edges:
            # If no valid edges, add self-loop to prevent errors
            valid_edges = [[0, 0]]
        
        return np.array(processed_features), np.array(valid_edges)

    def convert_to_pytorch_geometric(self, labels_df):
        data_list = []
        skipped_graphs = 0
        
        for apk_name, graph in self.graphs.items():
            try:
                # Skip if APK not in labels
                if apk_name not in labels_df.index:
                    print(f"Skipping {apk_name}: not found in labels")
                    skipped_graphs += 1
                    continue
                
                # Preprocess graph
                node_features, edges = self.preprocess_graph(graph)
                
                # Convert to PyTorch tensors
                x = torch.FloatTensor(node_features)
                edge_index = torch.LongTensor(edges).t()
                y = torch.LongTensor([labels_df.loc[apk_name, 'label']])
                
                # Create PyG data object
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)
                
            except Exception as e:
                print(f"Error processing {apk_name}: {str(e)}")
                skipped_graphs += 1
                continue
        
        print(f"Successfully processed {len(data_list)} graphs")
        print(f"Skipped {skipped_graphs} graphs due to errors")
        
        if not data_list:
            raise ValueError("No valid graphs were processed")
        
        return data_list

class GNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type='GCN'):
        super(GNNBlock, self).__init__()
        self.conv_type = conv_type
        
        if conv_type == 'GCN':
            self.conv = GCNConv(in_channels, out_channels)
        elif conv_type == 'GAT':
            self.conv = GATConv(in_channels, out_channels)
        elif conv_type == 'SAGE':
            self.conv = SAGEConv(in_channels, out_channels)
            
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class MalwareGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], conv_type='GCN'):
        super(MalwareGNN, self).__init__()
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # Input layer
        self.gnn_layers.append(GNNBlock(input_dim, hidden_dims[0], conv_type))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            self.gnn_layers.append(
                GNNBlock(hidden_dims[i], hidden_dims[i+1], conv_type)
            )
        
        # MLP for graph-level prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1]//2, hidden_dims[-1]//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1]//4, 2)  # Binary classification
        )
        
        # Layer normalization and attention
        self.layer_norm = nn.LayerNorm(hidden_dims[-1])
        self.attention = nn.Parameter(torch.randn(hidden_dims[-1], 1))
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Apply attention mechanism
        attention_weights = torch.tanh(torch.matmul(x, self.attention))
        attention_weights = F.softmax(attention_weights, dim=0)
        x = x * attention_weights
        
        # Graph pooling
        x = global_mean_pool(x, batch)
        
        # MLP classification
        out = self.mlp(x)
        return out

class MalwareDetector:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        print(f"Using device: {device}")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(data)
            loss = self.criterion(out, data.y)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, loader):
        self.model.eval()
        predictions = []
        labels = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
                pred = out.argmax(dim=1)
                
                predictions.extend(pred.cpu().numpy())
                labels.extend(data.y.cpu().numpy())
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1': f1_score(labels, predictions)
        }
    
    def train(self, train_loader, val_loader, epochs=100, patience=10):
        best_val_f1 = 0
        patience_counter = 0
        training_history = []
        
        for epoch in tqdm(range(epochs)):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Evaluation
            train_metrics = self.evaluate(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            # Save history
            history = {
                'epoch': epoch,
                'train_loss': train_loss,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            training_history.append(history)
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
            
            print(f'Epoch {epoch}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Train Metrics: {train_metrics}')
            print(f'Val Metrics: {val_metrics}')
            
        return pd.DataFrame(training_history)

def main():
    # Load data
    dataset = APKDataset('api_features.json')
    labels_df = pd.read_csv('sha256_family.csv', index_col='sha256')
    
    # Convert to PyG format
    data_list = dataset.convert_to_pytorch_geometric(labels_df)
    
    if len(data_list) < 3:  # Minimum required for train/val/test split
        raise ValueError("Not enough valid samples for training")
    
    # Split dataset
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    
    # Initialize model
    input_dim = data_list[0].x.shape[1]  # Feature dimension
    model = MalwareGNN(input_dim=input_dim, conv_type='GAT')
    
    # Train model
    detector = MalwareDetector(model)
    history = detector.train(train_loader, val_loader)
    
    # Save training history
    history.to_csv('training_history.csv', index=False)
    
    # Evaluate on test set
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = detector.evaluate(test_loader)
    print("Test Metrics:", test_metrics)
    
    # Save test metrics
    with open('test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

if __name__ == "__main__":
    main()