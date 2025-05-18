import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
import json
import os
import pickle

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

class OpCodeEmbedding:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.codebert = RobertaModel.from_pretrained('microsoft/codebert-base')
        self.opcode_mapping = {
            'move': ['move'],
            'add-double': ['add', 'double'],
            'const': ['const'],
            'float-to-int': ['float', 'to', 'int'],
            'if-eqz': ['if', 'equal', 'zero'],
            'iput-byte': ['instance', 'put', 'byte'],
            'return-void': ['return', 'void'],
            'invoke-virtual': ['invoke', 'virtual'],
            'new-instance': ['new', 'instance'],
            'if-eq': ['if', 'equal'],
            'if-ne': ['if', 'not', 'equal'],
            'if-gt': ['if', 'greater'],
            'if-ge': ['if', 'greater', 'equal'],
            'if-lt': ['if', 'less'],
            'if-le': ['if', 'less', 'equal'],
            'if-eqz': ['if', 'equal', 'zero'],
            'if-nez': ['if', 'not', 'equal', 'zero'],
            'if-lez': ['if', 'less', 'equal', 'zero'], 
            'if-ltz': ['if', 'less', 'zero'], 
            'if-gez': ['if', 'greater', 'equal', 'zero'],
            'if-gtz': ['if', 'greater', 'zero'],
        }
        # Initialize autoencoder (768 is CodeBERT's default output dimension, 64 is compressed dimension)
        self.autoencoder = Autoencoder(768, 64)
        # Load pretrained autoencoder if exists
        if os.path.exists('autoencoder.pth'):
            self.autoencoder.load_state_dict(torch.load('autoencoder.pth'))
        self.autoencoder.eval()

    def process_opcode_sequence(self, opcode_seq, compress=True):
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
            
            if compress:
                # Get CLS token embedding (first token) and compress it
                cls_embedding = embeddings[:, 0, :]
                compressed = self.autoencoder.encode(cls_embedding)
                return compressed
            
        return embeddings

def process_features(feature_file, output_folder, compress=True):
    """Process extracted features and create embeddings"""
    embedder = OpCodeEmbedding()
    
    with open(feature_file, 'r') as f:
        features = json.load(f)

    # Process opcodes for each method in the call graph
    for node in features['call_graph']['nodes']:
        opcodes = node[1].get('opcodes', [])
        if opcodes:  # Ensure there are opcodes to process
            node[1]['opcodes'] = embedder.process_opcode_sequence(opcodes, compress=compress).tolist()
    
    # Combine all features
    processed_features = {
        'permissions': features['permissions'],
        'call_graph': features['call_graph'],
        'label': features['label']
    }
    # Save processed features as .pkl
    base_name = os.path.basename(feature_file).replace('.json', '.pkl')
    output_path = os.path.join(output_folder, f"processed_{base_name}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_features, f)
    
    print(f"Saved compressed features to {output_path}")

def process_feature_folder(input_folder, output_folder, train_autoencoder_first=True):
    """Process all feature files in a folder"""
    os.makedirs(output_folder, exist_ok=True)
    
    for feature_file in os.listdir(input_folder):
        if not feature_file.endswith('.json'):
            continue

        base_name = os.path.basename(feature_file).replace('.json', '.pkl')
        output_path = os.path.join(output_folder, f"processed_{base_name}")
        if os.path.exists(output_path):
            print(f"File {output_path} already embedded. Skipping.")
            continue

        input_path = os.path.join(input_folder, feature_file)
        print(f"Processing features from: {input_path}")
        
        try:
            process_features(input_path, output_folder, compress=True)
        except Exception as e:
            print(f"Error processing {feature_file}: {str(e)}")

if __name__ == "__main__":
    feature_folder = "./test_extracted"
    processed_folder = "./test_embedded"
    
    process_feature_folder(feature_folder, processed_folder, train_autoencoder_first=True)