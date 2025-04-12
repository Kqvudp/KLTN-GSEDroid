import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SAGPooling
from torch_geometric.data import Data, DataLoader
import torch_geometric.nn
from torch.optim import Adam
import json
import os
from pathlib import Path
import pickle
from torch_geometric.loader import DataLoader  # Use the new loader

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, encoding_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, x):
        return self.decoder(x)

class TextCNNLayer(nn.Module):
    def __init__(self, embedding_dim=768, num_filters=64, filter_sizes=[2,3,4,5,6,7,8,9]):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs) 
            for fs in filter_sizes
        ])
        
        self.linear = nn.Linear(len(filter_sizes) * num_filters, 128)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        max_kernel_size = max([conv.kernel_size[0] for conv in self.convs])
        if x.size(2) < max_kernel_size:
            padding = max_kernel_size - x.size(2)
            x = F.pad(x, (0, padding))
        
        conv_results = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.shape[2])
            conv_results.append(pooled)
            
        x = torch.cat(conv_results, dim=1)
        x = x.squeeze(-1)
        
        x = self.linear(x)
        return x


class GSEDroidModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        
        # SAGEConv layers
        self.sage1 = SAGEConv(input_dim, hidden_dim, aggr='mean')
        self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.sage3 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.sage4 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        
        # SAGPooling layers
        self.pool1 = SAGPooling(hidden_dim)
        self.pool2 = SAGPooling(hidden_dim)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 2)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, batch):
        # Ensure batch tensor exists
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Graph convolution blocks
        x = F.relu(self.sage1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.sage2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = self.dropout(x)
        
        x = F.relu(self.sage3(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.sage4(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        
        # Global pooling and classification
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

def load_processed_features(feature_file, max_nodes=1000, autoencoder=None):
    """Load processed features from .pkl file and convert to PyTorch Geometric Data object"""
    with open(feature_file, 'rb') as f:
        features = pickle.load(f)
    
    # Convert method embeddings to node features
    nodes = []
    node_mapping = {}
    
    # First pass: collect nodes
    for idx, node in enumerate(features['call_graph']['nodes']):
        method_id = node[0]
        embedding = node[1].get('opcodes', [])
        if embedding:
            if idx >= max_nodes:
                break
                
            # If autoencoder is provided and we want to decode compressed embeddings
            if autoencoder is not None and hasattr(autoencoder, 'decode'):
                embedding_tensor = torch.tensor(embedding[0], dtype=torch.float).unsqueeze(0)
                with torch.no_grad():
                    decoded = autoencoder.decode(embedding_tensor)
                nodes.append(decoded.squeeze(0).tolist())
            else:
                nodes.append(embedding[0])  # Use the compressed embedding directly
                
            node_mapping[method_id] = idx
    
    num_nodes = len(nodes)
    if num_nodes == 0:
        print(f"Warning: No nodes found in {feature_file}, creating a default node.")
        embedding_dim = 128 if autoencoder is None else 768
        nodes = [[0.0] * embedding_dim]
        num_nodes = 1
    
    # Convert edges
    edges = []
    seen_edges = set()
    
    if 'edges' in features['call_graph']:
        for edge in features['call_graph']['edges']:
            if len(edge) == 2:
                source, target = edge
                if source in node_mapping and target in node_mapping:
                    source_idx = node_mapping[source]
                    target_idx = node_mapping[target]
                    
                    if source_idx < num_nodes and target_idx < num_nodes:
                        edge_tuple = (source_idx, target_idx)
                        if edge_tuple not in seen_edges:
                            edges.append(edge_tuple)
                            seen_edges.add(edge_tuple)
    
    # Filter out invalid edges
    edges = [(src, tgt) for src, tgt in edges if src < num_nodes and tgt < num_nodes]
    
    # Add self-loops if no edges exist
    if not edges:
        print(f"Warning: No valid edges found in {feature_file}, adding self-loops.")
        edges = [(i, i) for i in range(num_nodes)]
    
    # Convert to tensors
    x = torch.tensor(nodes, dtype=torch.float)
    
    # Convert edges to tensor and ensure proper format
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([features['label']], dtype=torch.long)
    
    # Final validation
    if edge_index.numel() > 0 and edge_index.max() >= x.size(0):
        print(f"Warning: edge_index contains invalid indices in {feature_file}.")
        edge_index = edge_index[:, edge_index[0] < x.size(0)]
        edge_index = edge_index[:, edge_index[1] < x.size(0)]
    
    # Create the data object with explicit batch assignment
    data = Data(x=x, edge_index=edge_index, y=y)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    
    return data

def create_dataloaders(feature_folder, batch_size=32, test_split=0.2, autoencoder_path=None):
    """Create train and test dataloaders from processed features in .pkl files"""
    graphs = []
    
    # Load autoencoder if path is provided
    autoencoder = None
    if autoencoder_path and os.path.exists(autoencoder_path):
        print(f"Loading autoencoder from {autoencoder_path}")
        autoencoder = Autoencoder(768, 64)  # Create model with original dimensions
        autoencoder.load_state_dict(torch.load(autoencoder_path))
        autoencoder.eval()
    
    for feature_file in os.listdir(feature_folder):
        if not feature_file.endswith('.pkl'):
            continue
            
        file_path = os.path.join(feature_folder, feature_file)
        try:
            data = load_processed_features(file_path, autoencoder=autoencoder)
            if data is not None and data.x.size(0) > 0 and data.edge_index.size(1) > 0:
                # Verify edge indices
                if data.edge_index.max() < data.x.size(0):
                    graphs.append(data)
                else:
                    print(f"Skipping {feature_file}: Invalid edge indices")
        except Exception as e:
            print(f"Error loading {feature_file}: {str(e)}")
    
    if not graphs:
        raise ValueError("No valid graphs were loaded!")
    
    # Split into train and test sets
    split_idx = max(1, int(len(graphs) * (1 - test_split)))
    train_data = graphs[:split_idx]
    test_data = graphs[split_idx:]
    
    print(f"\nTotal valid graphs loaded: {len(graphs)}")
    print(f"Training graphs: {len(train_data)}")
    print(f"Testing graphs: {len(test_data)}")
    
    # Print graph statistics
    max_nodes = max(g.x.size(0) for g in graphs)
    max_edges = max(g.edge_index.size(1) for g in graphs)
    feature_dim = graphs[0].x.size(1)
    print(f"Maximum nodes in any graph: {max_nodes}")
    print(f"Maximum edges in any graph: {max_edges}")
    print(f"Feature dimension: {feature_dim}")
    
    # Use the new DataLoader with follow_batch parameter
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, follow_batch=['x'])
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, follow_batch=['x'])
    
    return train_loader, test_loader

def train_model(model, train_loader, optimizer, device, epochs=50):
    model.train()
    training_stats = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            try:
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.nll_loss(out, batch.y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = out.max(1)[1]
                correct += pred.eq(batch.y).sum().item()
                total += batch.y.size(0)
                
            except RuntimeError as e:
                print(f"Error processing batch: {str(e)}")
                print(f"Batch statistics - Nodes: {batch.x.size(0)}, Edges: {batch.edge_index.size(1)}")
                continue
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total if total > 0 else 0
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Training Accuracy: {epoch_acc:.2f}%')
        
        training_stats.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'accuracy': epoch_acc
        })
    
    return training_stats

def save_model(model, path):
    """Save the model to disk"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    # Configuration
    FEATURE_FOLDER = "./test_embedded"
    MODEL_SAVE_PATH = "./trained_model.pth"
    AUTOENCODER_PATH = ""  # Path to pretrained autoencoder
    EPOCHS = 50
    BATCH_SIZE = 16
    HIDDEN_DIM = 64
    MAX_NODES = 1000
    LEARNING_RATE = 0.001
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders with autoencoder for decoding if available
    train_loader, test_loader = create_dataloaders(
        FEATURE_FOLDER, 
        BATCH_SIZE,
        autoencoder_path=AUTOENCODER_PATH
    )
    
    # Initialize model
    sample_data = next(iter(train_loader))
    input_dim = sample_data.x.size(1)
    model = GSEDroidModel(input_dim, HIDDEN_DIM).to(device)
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("Starting training...")
    training_stats = train_model(model, train_loader, optimizer, device, EPOCHS)
    
    # Save model
    save_model(model, MODEL_SAVE_PATH)