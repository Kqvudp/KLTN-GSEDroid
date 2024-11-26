import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SAGPooling
from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch.optim import Adam
import os
import torch_geometric.nn

class OpCodeProcessor:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.codebert = RobertaModel.from_pretrained('microsoft/codebert-base')
        self.text_cnn = TextCNNLayer()
        self.opcode_mapping = {
            'move': ['move'],
            'add-double': ['add', 'double'],
            'const': ['const'],
            'float-to-int': ['float', 'to', 'int'],
            'if-eqz': ['if', 'equal', 'zero'],
            'iput-byte': ['instance', 'put', 'byte'],
            'return-void': ['return', 'void'],
            'invoke-virtual': ['invoke', 'virtual']
        }

    def process_opcode_sequence(self, opcode_seq):
        tokens = []
        for opcode in opcode_seq:
            if opcode in self.opcode_mapping:
                tokens.extend(self.opcode_mapping[opcode])
            else:
                tokens.append(opcode)
                
        inputs = self.tokenizer(tokens, 
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=512)
        
        with torch.no_grad():
            outputs = self.codebert(**inputs)
            embeddings = outputs.last_hidden_state
            
        cnn_features = self.text_cnn(embeddings)
        return cnn_features

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

class APKAnalyzer:
    def __init__(self):
        self.opcode_processor = OpCodeProcessor()
        self.permission_list = self._get_top_permissions()
    
    def _get_top_permissions(self):
        return [
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
        
    def analyze_apk(self, apk_path):
        print(f"Analyzing APK: {apk_path}")
        a = apk.APK(apk_path)
        d = dvm.DalvikVMFormat(a.get_dex())
        dx = analysis.Analysis(d)
        
        permissions = self._extract_permissions(a)
        print(f"Permissions: {permissions}")
        
        call_graph = self._build_call_graph(dx)
        print(f"Call graph nodes: {len(call_graph.nodes)}, edges: {len(call_graph.edges)}")
        
        node_features = {}
        for method in dx.get_methods():
            if method.is_external():
                continue
                
            opcodes = [inst.get_name() for inst in method.get_instructions()]
            print(f"Opcodes for method {method.get_name()}: {opcodes}")
            
            embeddings = self.opcode_processor.process_opcode_sequence(opcodes)
            opcode_features = self.opcode_processor.text_cnn(embeddings)
            
            api_features = torch.cat([opcode_features, permissions], dim=-1)
            
            node_features[method] = api_features
            
        print(f"Node features: {len(node_features)}")
        return call_graph, node_features
        
    def _extract_permissions(self, apk_obj):
        perm_vector = torch.zeros(len(self.permission_list))
        
        apk_perms = apk_obj.get_permissions()
        
        for i, perm in enumerate(self.permission_list):
            if perm in apk_perms:
                perm_vector[i] = 1
                
        return perm_vector
        
    def _build_call_graph(self, analysis_obj):
        graph = nx.DiGraph()
        
        for method in analysis_obj.get_methods():
            if method.is_external():
                continue
                
            graph.add_node(method)
            
            for called in method.get_xref_to():
                if not called[1].is_external():
                    graph.add_edge(method, called[1])
                    
        return graph


def analyze_folder(folder_path, analyzer):
    graphs = []
    
    for apk_file in os.listdir(folder_path):
        apk_path = os.path.join(folder_path, apk_file)
        
        print(f"Processing {apk_path}...")
        
        graph, node_features = analyzer.analyze_apk(apk_path)
        
        nodes = torch.stack([features for _, features in node_features.items()])
        edges = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
        
        data = Data(x=nodes, edge_index=edges)
        graphs.append(data)
            
    return graphs

def create_dataloaders(graphs, batch_size=32, test_split=0.2):
    split_idx = int(len(graphs) * (1 - test_split))
    train_data = graphs[:split_idx]
    test_data = graphs[split_idx:]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    analyzer = APKAnalyzer()
    
    apk_folder = r"D:\FinalProject\code\test"
    
    print("Analyzing APK files...")
    graphs = analyze_folder(apk_folder, analyzer)
    
    print("Creating data loaders...")
    train_loader, test_loader = create_dataloaders(graphs, batch_size=32)
    
    print("Initializing model...")
    sample_data = next(iter(train_loader))
    input_dim = sample_data.x.size(1)
    
    model = GSEDroidModel(input_dim=input_dim).to(device)
    
    print("Setting up optimizer...")
    optimizer = Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    train_model(model, train_loader, optimizer, epochs=50)
    
    print("Evaluating model...")
    accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

def to_device(batch, device):
    batch.x = batch.x.to(device)
    batch.edge_index = batch.edge_index.to(device)
    batch.batch = batch.batch.to(device)
    batch.y = batch.y.to(device)
    return batch

def train_model(model, train_loader, optimizer, epochs=50):
    device = next(model.parameters()).device
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.nll_loss(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

def evaluate(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch, device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)
            
    return correct / total