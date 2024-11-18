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

class APIFeatureExtractor:
    def __init__(self, max_seq_length=512):
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.codebert = RobertaModel.from_pretrained('microsoft/codebert-base')
        self.textcnn = TextCNN()
        self.max_seq_length = max_seq_length
        self.third_party_prefixes = [
            'Lorg/codehaus/', 'Lcom/apperhand/', 'Lcom/airpush/', 'Lcom/mobclix/'
        ]
        
    def get_permission_vector(self, apk):
        """Convert permissions to binary vector"""
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
        permission_vector = [1 if p in app_permissions else 0 for p in all_permissions]
        return permission_vector
    
    def get_opcode_embedding(self, method):
        """Get opcode sequence embedding using CodeBERT and TextCNN"""
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

    def is_third_party_api(self, method_name):
        """Check if the method is from a third-party library"""
        return any(method_name.startswith(prefix) for prefix in self.third_party_prefixes)
    
    def extract_api_features(self, apk_path):
        """Extract API features from APK with filtering"""
        # Load APK
        apk, dex_list, dx = AnalyzeAPK(apk_path)
        
        # Initialize graph and tracking sets
        init_graph = nx.DiGraph()
        node_vectors = {}
        methods_calling_apis = set()
        methods_called_by_apis = set()
        
        # Get permission vector
        per_vec = self.get_permission_vector(apk)
        
        # First pass: Identify API calls and build initial graph
        for dex in dex_list:
            for method in dex.get_methods():
                method_name = method.get_name()
                
                if method.get_code() and not self.is_third_party_api(method_name):
                    for instruction in method.get_code().get_bc().get_instructions():
                        if instruction.get_name().startswith('invoke-'):
                            called_method = instruction.get_output().split(',')[-1].strip()
                            
                            if not self.is_third_party_api(called_method):
                                methods_calling_apis.add(method_name)
                                methods_called_by_apis.add(called_method)
                                init_graph.add_edge(method_name, called_method)
        
        # Second pass: Process only non-isolated nodes
        connected_nodes = set()
        for node in init_graph.nodes():
            if init_graph.degree(node) > 0:  # Node has at least one edge
                connected_nodes.add(node)
        
        # Create final graph with features for non-isolated nodes
        final_graph = {
            'nodes': [],
            'edges': []
        }
        
        # Process nodes
        for dex in dex_list:
            for method in dex.get_methods():
                method_name = method.get_name()
                
                if method_name in connected_nodes:
                    op_vec = self.get_opcode_embedding(method)
                    if op_vec is not None:
                        # Combine opcode and permission vectors
                        api_vec = np.concatenate([op_vec, per_vec])
                        final_graph['nodes'].append({
                            'id': method_name,
                            'features': api_vec.tolist()
                        })
        
        # Add edges for connected nodes
        for u, v in init_graph.edges():
            if u in connected_nodes and v in connected_nodes:
                final_graph['edges'].append({
                    'source': u,
                    'target': v
                })
        
        return final_graph

def process_apk_batch(apk_paths, output_path):
    """Process multiple APK files and save results to JSON"""
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
    
    # Save results to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

# Example usage
if __name__ == "__main__":
    apk_directory = r"/kaggle/input/apkmalwarerawfile"
    output_path = "/kaggle/working/api_features.json"
    
    # Get list of APK files
    apk_paths = [
        os.path.join(apk_directory, f) 
        for f in os.listdir(apk_directory)
    ]
    
    # Process APKs and save features
    process_apk_batch(apk_paths, output_path)