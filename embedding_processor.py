import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
import json
import os
from pathlib import Path

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

class OpCodeEmbedding:
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
                parts = opcode.split('/')[0]
                if parts in self.opcode_mapping:
                    tokens.extend(self.opcode_mapping[parts])
                else:
                    split_parts = parts.split('-')
                    tokens.extend(split_parts)
                
        inputs = self.tokenizer(' '.join(tokens), 
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=512)
        
        with torch.no_grad():
            outputs = self.codebert(**inputs)
            embeddings = outputs.last_hidden_state
            
        return self.text_cnn(embeddings)

def process_features(feature_file, output_folder):
    """Process extracted features and create embeddings"""
    embedder = OpCodeEmbedding()
    
    with open(feature_file, 'r') as f:
        features = json.load(f)
    
    # Process opcodes for each method
    method_embeddings = {}
    for method_id, opcodes in features['opcodes'].items():
        method_embeddings[method_id] = embedder.process_opcode_sequence(opcodes).tolist()
    
    # Combine all features
    processed_features = {
        'method_embeddings': method_embeddings,
        'permissions': features['permissions'],
        'call_graph': features['call_graph'],
        'label': features['label']
    }
    
    # Save processed features
    base_name = os.path.basename(feature_file)
    output_path = os.path.join(output_folder, f"processed_{base_name}")
    
    with open(output_path, 'w') as f:
        json.dump(processed_features, f, indent=2)
    
    print(f"Saved processed features to {output_path}")

def process_feature_folder(input_folder, output_folder):
    """Process all feature files in a folder"""
    os.makedirs(output_folder, exist_ok=True)
    
    for feature_file in os.listdir(input_folder):
        if not feature_file.endswith('.json'):
            continue
            
        input_path = os.path.join(input_folder, feature_file)
        print(f"Processing features from: {input_path}")
        
        try:
            process_features(input_path, output_folder)
        except Exception as e:
            print(f"Error processing {feature_file}: {str(e)}")

if __name__ == "__main__":
    feature_folder = "path/to/features"
    processed_folder = "path/to/processed_features"
    
    process_feature_folder(feature_folder, processed_folder)