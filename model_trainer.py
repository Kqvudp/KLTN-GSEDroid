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

class GSEDroidModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.sage3 = SAGEConv(hidden_dim, hidden_dim)
        self.sage4 = SAGEConv(hidden_dim, hidden_dim)
        
        self.pool1 = SAGPooling(hidden_dim)
        self.pool2 = SAGPooling(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 2)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.sage1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.sage2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = self.dropout(x)
        
        x = F.relu(self.sage3(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.sage4(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        
        x = torch_geometric.nn.global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
def load_processed_features(feature_file):
    """Load processed features and convert to PyTorch Geometric Data object"""
    with open(feature_file, 'r') as f:
        features = json.load(f)
    
    # Convert method embeddings to node features
    nodes = []
    node_mapping = {}
    method_list = []  # Keep track of original method IDs
    
    for method_id, embedding in features['method_embeddings'].items():
        nodes.append(embedding)
        node_mapping[method_id] = len(nodes) - 1  # Use current length as index
        method_list.append(method_id)
    
    if not nodes:
        embedding_dim = 128
        nodes = [[0.0] * embedding_dim]
    
    # Convert edges to indices after ensuring all nodes are mapped
    edges = []
    for source, target in features['call_graph']['edges']:
        # Only use edges where both nodes exist in our mapping
        if source in node_mapping and target in node_mapping:
            source_idx = node_mapping[source]
            target_idx = node_mapping[target]
            edges.append([source_idx, target_idx])
    
    if not edges:
        edges = [[0, 0]]  # Add self-loop for single node
    
    # Convert to tensors
    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([features['label']], dtype=torch.long)
    
    # Validate edge indices
    max_index = edge_index.max().item()
    num_nodes = x.size(0)
    
    if max_index >= num_nodes:
        print(f"Warning: Invalid graph in {feature_file}")
        print(f"Nodes: {num_nodes}, Max edge index: {max_index}")
        # Filter invalid edges
        valid_edges = []
        for i in range(edge_index.size(1)):
            source = edge_index[0, i].item()
            target = edge_index[1, i].item()
            if source < num_nodes and target < num_nodes:
                valid_edges.append([source, target])
        
        if not valid_edges:
            valid_edges = [[0, 0]]  # Fallback to self-loop
        edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
    
    # Final validation
    print(f"Processed {feature_file}:")
    print(f"Number of nodes: {x.size(0)}")
    print(f"Number of edges: {edge_index.size(1)}")
    print(f"Edge index range: [0, {edge_index.max().item()}]")
    print("-" * 50)
    
    return Data(x=x, edge_index=edge_index, y=y)

def create_dataloaders(feature_folder, batch_size=32, test_split=0.2):
    """Create train and test dataloaders from processed features"""
    graphs = []
    
    for feature_file in os.listdir(feature_folder):
        if not feature_file.startswith('processed_') or not feature_file.endswith('.json'):
            continue
            
        file_path = os.path.join(feature_folder, feature_file)
        try:
            data = load_processed_features(file_path)
            # Validate graph before adding
            if (data.x.size(0) > 0 and 
                data.edge_index.size(1) > 0 and 
                data.edge_index.max() < data.x.size(0)):
                graphs.append(data)
            else:
                print(f"Skipping invalid graph from {feature_file}")
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
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, optimizer, device, epochs=50):
    """Train the model and return training statistics"""
    model.train()
    training_stats = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.nll_loss(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.max(1)[1]
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Training Accuracy: {epoch_acc:.2f}%')
        
        training_stats.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'accuracy': epoch_acc
        })
    
    return training_stats

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.nll_loss(out, batch.y)
            total_loss += loss.item()
            
            pred = out.max(1)[1]
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)
    
    test_loss = total_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    
    return test_loss, test_accuracy

def save_model(model, save_path):
    """Save the trained model"""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path):
    """Load a trained model"""
    model.load_state_dict(torch.load(load_path))
    print(f"Model loaded from {load_path}")
    return model

if __name__ == "__main__":
    # Configuration
    FEATURE_FOLDER = r"D:\FinalProject\code\main_v4\processed_features"
    MODEL_SAVE_PATH = r"D:\FinalProject\code\main_v4\trained_model.pth"
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 128
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(FEATURE_FOLDER, BATCH_SIZE)
    
    # Initialize model
    sample_data = next(iter(train_loader))
    input_dim = sample_data.x.size(1)
    model = GSEDroidModel(input_dim, HIDDEN_DIM).to(device)
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("Starting training...")
    training_stats = train_model(model, train_loader, optimizer, device, EPOCHS)
    
    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, test_loader, device)
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.2f}%")
    
    # Save model
    save_model(model, MODEL_SAVE_PATH)