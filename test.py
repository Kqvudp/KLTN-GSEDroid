import json
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from androguard.core.bytecodes.apk import APK
from androguard.core.analysis.analysis import Analysis
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.misc import AnalyzeAPK
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

class TextCNN(nn.Module):
    def __init__(self, embed_dim=768, kernel_sizes=[2,3,4,5,6,7,8,9], num_filters=128):
        super(TextCNN, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, 128)
        
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        
        # Apply convolution and max-pooling
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, seq_len-k+1)
            pooled = F.max_pool1d(conv_out, conv_out.shape[2])  # (batch_size, num_filters, 1)
            pooled_outputs.append(pooled)
        
        # Concatenate pooled outputs
        pooled_concat = torch.cat(pooled_outputs, dim=1)  # (batch_size, num_filters * len(kernel_sizes), 1)
        pooled_concat = pooled_concat.squeeze(-1)  # (batch_size, num_filters * len(kernel_sizes))
        
        # Apply dropout and fully connected layer
        out = self.dropout(pooled_concat)
        out = self.fc(out)
        return out

class APIFeatureExtractor:
    def __init__(self, max_seq_length=512):
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.codebert = RobertaModel.from_pretrained('microsoft/codebert-base')
        self.textcnn = TextCNN()
        self.max_seq_length = max_seq_length
        
    def get_permission_vector(self, apk):
        """Convert permissions to binary vector"""
        all_permissions = set()
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
            
        # Extract opcode sequence
        opcodes = [ins.get_name() for ins in method.get_code().get_bc().get_instructions()]
        opcode_text = ' '.join(opcodes)
        
        # Tokenize and encode
        inputs = self.tokenizer(opcode_text, 
                              padding='max_length',
                              max_length=self.max_seq_length,
                              truncation=True,
                              return_tensors='pt')
        
        # Get CodeBERT embeddings
        with torch.no_grad():
            outputs = self.codebert(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Apply TextCNN
        cnn_output = self.textcnn(embeddings)
        return cnn_output.squeeze(0).detach().numpy()
    
    def extract_api_features(self, apk_path):
        """Extract API features from APK following the algorithm"""
        # Load APK
        apk, dex_list, dx = AnalyzeAPK(apk_path)
        
        # Initialize graph
        init_graph = nx.DiGraph()
        node_vectors = {}
        
        # Get permission vector
        per_vec = self.get_permission_vector(apk)
        
        # Process each DEX file
        for dex in dex_list:
            for method in dex.get_methods():
                method_name = method.get_name()
                
                # Get opcode embedding
                op_vec = self.get_opcode_embedding(method)
                if op_vec is None:
                    continue
                
                # Add node to graph
                if method.get_code():
                    init_graph.add_node(method_name)
                    
                    # Extract API calls and build edges
                    for instruction in method.get_code().get_bc().get_instructions():
                        if instruction.get_name().startswith('invoke-'):
                            called_method = instruction.get_output().split(',')[-1].strip()
                            init_graph.add_edge(method_name, called_method)
                            
                            # Combine opcode and permission vectors
                            api_vec = np.concatenate([op_vec, per_vec])
                            node_vectors[method_name] = api_vec.tolist()
        
        # Create final graph with features
        final_graph = {
            'nodes': [{
                'id': node,
                'features': node_vectors.get(node, [])
            } for node in init_graph.nodes()],
            'edges': [{
                'source': u,
                'target': v
            } for u, v in init_graph.edges()]
        }
        
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
    apk_directory = r"E:\drebin\drebin-cut"
    output_path = "api_features.json"
    
    # Get list of APK files
    apk_paths = [
        os.path.join(apk_directory, f) 
        for f in os.listdir(apk_directory)
    ]
    
    # Process APKs and save features
    process_apk_batch(apk_paths, output_path)