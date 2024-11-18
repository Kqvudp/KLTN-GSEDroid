import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from androguard.core.bytecodes.apk import APK
from androguard.core.analysis.analysis import Analysis
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.misc import AnalyzeAPK
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

class TextCNN(nn.Module):
    def __init__(self, embed_dim=768, kernel_sizes=[2,3,4,5,6,7,8,9], num_filters=128):
        super(TextCNN, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, 128)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.shape[2])
            pooled_outputs.append(pooled)
        
        pooled_concat = torch.cat(pooled_outputs, dim=1)
        pooled_concat = pooled_concat.squeeze(-1)
        
        out = self.dropout(pooled_concat)
        out = self.fc(out)
        return out

class GraphReduction:
    """Handles graph reduction operations as shown in the flowchart"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def reduce_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Reduces graph complexity while preserving important structural information"""
        # Calculate node importance using centrality metrics
        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        
        # Combine metrics
        node_importance = {
            node: (degree_centrality[node] + betweenness_centrality[node])/2
            for node in graph.nodes()
        }
        
        # Keep only important nodes (top 70%)
        threshold = np.percentile(list(node_importance.values()), 30)
        important_nodes = [
            node for node, importance in node_importance.items()
            if importance >= threshold
        ]
        
        # Create reduced graph
        reduced_graph = graph.subgraph(important_nodes).copy()
        return reduced_graph

class APIFeatureExtractor:
    def __init__(self, max_seq_length=512):
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.codebert = RobertaModel.from_pretrained('microsoft/codebert-base')
        self.textcnn = TextCNN()
        self.max_seq_length = max_seq_length
        self.graph_reducer = GraphReduction()
        
    def get_permission_vector(self, apk: APK) -> List[int]:
        """Extract and vectorize permissions"""
        all_permissions = [
            'android.permission.INTERNET',
            'android.permission.READ_PHONE_STATE',
            'android.permission.SEND_SMS',
            'android.permission.WRITE_EXTERNAL_STORAGE',
            'android.permission.ACCESS_NETWORK_STATE',
            'android.permission.RECEIVE_SMS',
            'android.permission.RECEIVE_BOOT_COMPLETED',
            'android.permission.READ_SMS',
            'android.permission.ACCESS_WIFI_STATE',
            'android.permission.ACCESS_COARSE_LOCATION',
            'android.permission.VIBRATE',
            'android.permission.WRITE_SMS',
            'android.permission.WAKE_LOCK',
            'android.permission.ACCESS_FINE_LOCATION',
            'android.permission.INSTALL_SHORTCUT',
            'android.permission.READ_CONTACTS',
            'android.permission.INSTALL_PACKAGES',
            'android.permission.CALL_PHONE',
            'android.permission.GET_TASKS',
            'android.permission.SET_WALLPAPER',
            'android.permission.READ_EXTERNAL_STORAGE',
            'android.permission.WRITE_EXTERNAL_STORAGE'
        ]
        
        app_permissions = set(p for p in apk.get_permissions() if p.startswith('android.permission.'))
        return [1 if p in app_permissions else 0 for p in all_permissions]
    
    def get_opcode_embedding(self, method) -> np.ndarray:
        """Generate embeddings for opcode sequences using CodeBERT and TextCNN"""
        if not method.get_code():
            return None
            
        opcodes = [ins.get_name() for ins in method.get_code().get_bc().get_instructions()]
        opcode_text = ' '.join(opcodes)
        
        inputs = self.tokenizer(opcode_text, 
                              padding='max_length',
                              max_length=self.max_seq_length,
                              truncation=True,
                              return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.codebert(**inputs)
            embeddings = outputs.last_hidden_state
        
        cnn_output = self.textcnn(embeddings)
        return cnn_output.squeeze(0).detach().numpy()
    
    def build_api_call_graph(self, dex_list) -> Tuple[nx.DiGraph, Dict]:
        """Construct initial API call graph"""
        graph = nx.DiGraph()
        node_features = {}
        
        for dex in dex_list:
            for method in dex.get_methods():
                method_name = method.get_name()
                if method.get_code():
                    graph.add_node(method_name)
                    
                    # Extract API calls and build edges
                    for instruction in method.get_code().get_bc().get_instructions():
                        if instruction.get_name().startswith('invoke-'):
                            called_method = instruction.get_output().split(',')[-1].strip()
                            graph.add_edge(method_name, called_method)
                            
                            # Store opcode embeddings
                            op_vec = self.get_opcode_embedding(method)
                            if op_vec is not None:
                                node_features[method_name] = op_vec
        
        return graph, node_features
    
    def extract_api_features(self, apk_path: str) -> Dict:
        """Main feature extraction pipeline following the flowchart"""
        # Load APK
        apk, dex_list, dx = AnalyzeAPK(apk_path)
        
        # Extract permissions
        permission_vector = self.get_permission_vector(apk)
        
        # Build initial API call graph
        initial_graph, node_features = self.build_api_call_graph(dex_list)
        
        # Apply graph reduction
        reduced_graph = self.graph_reducer.reduce_graph(initial_graph)
        
        # Combine features for final graph
        final_graph = {
            'nodes': [{
                'id': node,
                'features': np.concatenate([
                    node_features.get(node, np.zeros(128)),  # opcode embeddings
                    permission_vector  # permission vector
                ]).tolist() if node in node_features else []
            } for node in reduced_graph.nodes()],
            'edges': [{
                'source': u,
                'target': v
            } for u, v in reduced_graph.edges()]
        }
        
        return final_graph

def process_apk_batch(apk_paths: List[str], output_path: str):
    """Process multiple APK files in parallel"""
    extractor = APIFeatureExtractor()
    results = {}
    
    with ThreadPoolExecutor() as executor:
        future_to_apk = {
            executor.submit(extractor.extract_api_features, apk_path): apk_path 
            for apk_path in apk_paths
        }
        
        for future in as_completed(future_to_apk):
            apk_path = future_to_apk[future]
            try:
                graph = future.result()
                results[os.path.basename(apk_path)] = graph
            except Exception as e:
                print(f"Error processing {apk_path}: {str(e)}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    apk_directory = r"D:\FinalProject\code"
    output_path = "api_features.json"
    
    apk_paths = [
        os.path.join(apk_directory, f) 
        for f in os.listdir(apk_directory)
    ]
    
    process_apk_batch(apk_paths, output_path)