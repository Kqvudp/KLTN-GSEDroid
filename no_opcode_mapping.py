import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SAGPooling
from torch_geometric.data import Data
import androguard.core.bytecodes.dvm as dvm
from androguard.core.bytecodes.apk import APK
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from torch_geometric.data import DataLoader
import os

class APIFeatureExtractor:
    def __init__(self):
        # Initialize CodeBERT tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.codebert = RobertaModel.from_pretrained('microsoft/codebert-base')
        self.linear = nn.Linear(512, 128)
        # Top permissions to track based on frequency
        self.top_permissions = [
            'android.permission.INTERNET',
            'android.permission.ACCESS_NETWORK_STATE', 
            'android.permission.WRITE_EXTERNAL_STORAGE',
            'android.permission.ACCESS_WIFI_STATE',
            'android.permission.READ_PHONE_STATE',
            'android.permission.WAKE_LOCK',
            'android.permission.VIBRATE',
            'android.permission.READ_EXTERNAL_STORAGE',
            'android.permission.ACCESS_FINE_LOCATION',
            'android.permission.ACCESS_COARSE_LOCATION'
        ]

    def get_permission_vector(self, apk):
        """Extract permission features from APK"""
        permissions = apk.get_permissions()
        return torch.tensor([1 if p in permissions else 0 for p in self.top_permissions])

    def process_opcode_sequence(self, method):
        """Process opcode sequence using CodeBERT and TextCNN"""
        # Get opcode sequence from method
        opcodes = [i.get_name() for i in method.get_instructions()]
        
        # Tokenize opcodes
        tokens = self.tokenizer(" ".join(opcodes), padding=True, truncation=True, 
                              max_length=512, return_tensors="pt")
        
        # Get CodeBERT embeddings
        with torch.no_grad():
            outputs = self.codebert(**tokens)
            embeddings = outputs.last_hidden_state
        
        # TextCNN processing
        conv_layers = []
        for kernel_size in range(2, 10):
            conv = nn.Conv1d(768, 64, kernel_size, stride=1, padding=kernel_size//2)
            conv_out = conv(embeddings.transpose(1, 2))
            pooled = F.max_pool1d(conv_out, conv_out.shape[2])
            conv_layers.append(pooled)
            
        concat = torch.cat(conv_layers, dim=1)
        return self.linear(concat.squeeze())

class GSEDroid(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(GSEDroid, self).__init__()
        
        # Graph Sage layers
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, hidden_dim)
        
        # SAG Pooling layers
        self.pool1 = SAGPooling(hidden_dim, ratio=0.8)
        self.pool2 = SAGPooling(hidden_dim, ratio=0.8)
        
        # Final classification layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 2)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        # First pooling
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, data.batch)
        
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.dropout(x)
        
        # Second pooling
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        
        # Global mean pooling
        x = torch.mean(x, dim=0)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=-1)

def process_apk(apk_path):
    """Process an APK file and create graph data"""
    try:
        feature_extractor = APIFeatureExtractor()
        apk = APK(apk_path)
        
        # Get DEX analysis
        from androguard.core.bytecodes.dvm import DalvikVMFormat
        from androguard.core.analysis.analysis import Analysis
        
        dex = DalvikVMFormat(apk.get_dex())
        dx = Analysis(dex)
        dx.create_xref()  # Create cross references
        
        # Extract API calls and build graph
        api_calls = []
        edge_list = []
        features = []
        
        for method in dex.get_methods():
            # Get method features
            permission_vec = feature_extractor.get_permission_vector(apk)
            opcode_vec = feature_extractor.process_opcode_sequence(method)
            features.append(torch.cat([permission_vec, opcode_vec]))
            
            # Add edges based on method calls
            for _, call, _ in method.get_xref_to():  # Use correct method to get xrefs
                edge_list.append([len(api_calls), len(api_calls) + 1])
            api_calls.append(method)
        
        if not features:
            raise ValueError(f"No features extracted from {apk_path}")
            
        # Create PyTorch Geometric data object
        x = torch.stack(features)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        
        return data
        
    except Exception as e:
        print(f"Error processing {apk_path}: {str(e)}")
        return None

def train_model(model, train_loader, optimizer, epochs=50):
    """Train the GSEDroid model"""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

def evaluate_model(model, test_loader):
    """Evaluate the GSEDroid model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)
    
    return correct / total


def create_train_loader(apk_folder, batch_size=32, malware_folder="malware", benign_folder="benign"):
    dataset = []
    
    # Process malware samples
    malware_path = os.path.join(apk_folder, malware_folder)
    for apk_file in os.listdir(malware_path):
        if apk_file.endswith('.apk'):
            data = process_apk(os.path.join(malware_path, apk_file))
            if data is not None:
                data.y = torch.tensor(1)  # Label 1 for malware
                dataset.append(data)
    
    # Process benign samples 
    benign_path = os.path.join(apk_folder, benign_folder)
    for apk_file in os.listdir(benign_path):
        if apk_file.endswith('.apk'):
            data = process_apk(os.path.join(benign_path, apk_file))
            if data is not None:
                data.y = torch.tensor(0)  # Label 0 for benign
                dataset.append(data)
    
    if not dataset:
        raise ValueError("No valid APK files processed")
        
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader

# Usage example:
train_loader = create_train_loader(
    apk_folder="D:\FinalProject\code\main",
    batch_size=32
)

model = GSEDroid(input_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, optimizer)