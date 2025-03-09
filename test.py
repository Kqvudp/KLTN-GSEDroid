import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import os
from androguard.misc import AnalyzeAPK
from collections import defaultdict

class SeGDroid:
    def __init__(self, sensitive_api_file="sensitive_apis.txt", embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.sensitive_apis = self._load_sensitive_apis(sensitive_api_file)
        self.api2vec = None
        self.opcode2vec = None
        self.model = None
        
    def _load_sensitive_apis(self, file_path):
        """Load sensitive APIs from file or use a small default set if file doesn't exist"""
        try:
            with open(file_path, 'r') as f:
                return set([line.strip() for line in f])
        except FileNotFoundError:
            # Default small set of sensitive APIs for demonstration
            print(f"Warning: {file_path} not found. Using default sensitive API set.")
            return {
                "android.permission.READ_CONTACTS",
                "android.permission.SEND_SMS",
                "android.permission.READ_SMS",
                "android.permission.ACCESS_FINE_LOCATION",
                "android.telephony.SmsManager",
                "android.telephony.TelephonyManager.getDeviceId",
                "java.net.HttpURLConnection"
            }
            
    def extract_fcg(self, apk_path):
        """Extract function call graph from APK"""
        try:
            a, d, dx = AnalyzeAPK(apk_path)
            dx.create_xref()
            
            # Build the function call graph
            fcg = nx.DiGraph()
            
            # Extract method calls
            for method in dx.get_methods():
                # method_name = method.get_name()
                # class_name = method.get_class_name()
                # full_method = f"{class_name}->{method_name}"
                
                fcg.add_node(method, 
                            is_external=False, 
                            opcodes=self._get_opcodes(method),
                            api="")
                
                # Add edges for method calls
                for xref, xref_type in method.get_xref_to():
                    if xref_type == 'INVOKE':
                        callee = f"{xref.get_class_name()}->{xref.get_name()}"
                        fcg.add_edge(method, callee)
                        
                        # Mark external nodes and add API info
                        if "android" in callee or "java" in callee:
                            fcg.nodes[callee]['is_external'] = True
                            fcg.nodes[callee]['api'] = callee
                            fcg.nodes[callee]['opcodes'] = []
            
            return fcg
        except Exception as e:
            print(f"Error extracting FCG from {apk_path}: {e}")
            return nx.DiGraph()
    
    def _get_opcodes(self, method):
        """Extract opcodes from a method"""
        try:
            opcodes = []
            if method.get_code():
                for instruction in method.get_code().get_bc().get_instructions():
                    opcodes.append(instruction.get_name())
            return opcodes
        except:
            return []
            
    def prune_graph(self, fcg, threshold=8000):
        """Prune FCG using sensitive API-based approach"""
        # Skip pruning for small graphs
        if len(fcg.nodes) <= threshold:
            return fcg
            
        # Find sensitive nodes
        sensitive_nodes = set()
        for node, attrs in fcg.nodes(data=True):
            if attrs.get('api') and any(api in attrs['api'] for api in self.sensitive_apis):
                sensitive_nodes.add(node)
                
        if not sensitive_nodes:
            return fcg  # No sensitive nodes found
                
        # Find ancestor nodes (upward direction)
        ancestor_nodes = set()
        for sv in sensitive_nodes:
            # Use depth-first search to find ancestors
            ancestors = nx.ancestors(fcg, sv)
            ancestor_nodes.update(ancestors)
            ancestor_nodes.add(sv)
                
        # Find descendant nodes (downward direction)
        descendant_nodes = set()
        for sav in ancestor_nodes:
            # Use depth-first search to find descendants
            descendants = nx.descendants(fcg, sav)
            descendant_nodes.update(descendants)
            descendant_nodes.add(sav)
                
        # Create the pruned graph
        all_preserved_nodes = ancestor_nodes.union(descendant_nodes)
        pruned_fcg = fcg.subgraph(all_preserved_nodes).copy()
        
        return pruned_fcg
    
    def train_embeddings(self, apk_paths, corpus_dir="corpus"):
        """Train API2vec and opcode2vec models"""
        # Create corpus directory if it doesn't exist
        os.makedirs(corpus_dir, exist_ok=True)
        
        # Create corpus files
        api_corpus_file = os.path.join(corpus_dir, "api_corpus.txt")
        opcode_corpus_file = os.path.join(corpus_dir, "opcode_corpus.txt")
        
        with open(api_corpus_file, 'w') as api_file, open(opcode_corpus_file, 'w') as opcode_file:
            for apk_path in apk_paths:
                fcg = self.extract_fcg(apk_path)
                
                # Extract API packages and opcode sequences
                for node, attrs in fcg.nodes(data=True):
                    if attrs.get('is_external', False) and attrs.get('api'):
                        # Write API package parts to corpus
                        parts = attrs['api'].split('.')
                        api_file.write(' '.join(parts) + '\n')
                    elif attrs.get('opcodes'):
                        # Write opcode sequence to corpus
                        opcode_file.write(' '.join(attrs['opcodes']) + '\n')
        
        # Train word2vec models
        self.api2vec = Word2Vec(
            LineSentence(api_corpus_file),
            vector_size=self.embedding_dim // 2,
            window=5,
            min_count=1,
            sg=1,  # Use skip-gram
            workers=4
        )
        
        self.opcode2vec = Word2Vec(
            LineSentence(opcode_corpus_file),
            vector_size=self.embedding_dim // 2,
            window=5,
            min_count=1,
            sg=1,  # Use skip-gram
            workers=4
        )
        
        print("Embedding models trained successfully")
    
    def create_node_vectors(self, fcg):
        """Create feature vectors for nodes using embeddings and centrality"""
        # Check if embeddings are trained
        if self.api2vec is None or self.opcode2vec is None:
            raise ValueError("Embeddings must be trained before creating node vectors")
        
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(fcg)
        
        node_vectors = {}
        for node, attrs in fcg.nodes(data=True):
            # Initialize vectors
            api_vector = np.zeros(self.embedding_dim // 2)
            opcode_vector = np.zeros(self.embedding_dim // 2)
            
            # Get API embeddings for external nodes
            if attrs.get('is_external', False) and attrs.get('api'):
                parts = attrs['api'].split('.')
                part_vectors = []
                for part in parts:
                    try:
                        part_vectors.append(self.api2vec.wv[part])
                    except KeyError:
                        continue
                if part_vectors:
                    api_vector = np.mean(part_vectors, axis=0)
            
            # Get opcode embeddings for internal nodes
            elif attrs.get('opcodes'):
                opcode_vecs = []
                for opcode in attrs['opcodes']:
                    try:
                        opcode_vecs.append(self.opcode2vec.wv[opcode])
                    except KeyError:
                        continue
                if opcode_vecs:
                    opcode_vector = np.mean(opcode_vecs, axis=0)
            
            # Concatenate vectors
            node_vector = np.concatenate([api_vector, opcode_vector])
            
            # Weight by centrality
            centrality = degree_centrality.get(node, 0)
            weighted_vector = node_vector * centrality
            
            node_vectors[node] = weighted_vector
            
        return node_vectors
    
    def prepare_data_for_gnn(self, fcgs, labels):
        """Convert FCGs to PyTorch Geometric Data objects"""
        data_list = []
        
        for i, fcg in enumerate(fcgs):
            if len(fcg.nodes) == 0:
                continue
                
            # Create node feature matrix
            node_vectors = self.create_node_vectors(fcg)
            node_list = list(fcg.nodes())
            node_features = np.array([node_vectors[node] for node in node_list])
            
            # Create edge index
            edge_list = list(fcg.edges())
            edge_index = []
            for src, dst in edge_list:
                src_idx = node_list.index(src)
                dst_idx = node_list.index(dst)
                edge_index.append([src_idx, dst_idx])
                
            if not edge_index:
                continue
                
            edge_index = np.array(edge_index).T
            
            # Convert to torch tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            y = torch.tensor([labels[i]], dtype=torch.long)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
            
        return data_list

class GraphSAGENet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Fully connected layer
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_importance(self, data):
        """Calculate node importance for model explanation"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Get node embeddings from the last conv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Get weights from the final linear layer
        weights = self.fc.weight.detach().cpu().numpy()
        
        # Calculate importance for each node
        node_embeddings = x.detach().cpu().numpy()
        node_importance = np.dot(node_embeddings, weights.T)
        
        return node_importance

def train_segdroid(apk_paths, labels, test_apk_paths, test_labels, embedding_dim=64, epochs=50):
    """Train and evaluate a SeGDroid model"""
    # Initialize SeGDroid
    segdroid = SeGDroid(embedding_dim=embedding_dim)
    
    # Train word embeddings
    print("Training word embeddings...")
    segdroid.train_embeddings(apk_paths)
    
    # Extract and prune FCGs
    print("Extracting and pruning FCGs...")
    fcgs = []
    for apk_path in apk_paths:
        fcg = segdroid.extract_fcg(apk_path)
        pruned_fcg = segdroid.prune_graph(fcg)
        fcgs.append(pruned_fcg)
    
    # Prepare data for GNN
    print("Preparing data for GNN...")
    train_data_list = segdroid.prepare_data_for_gnn(fcgs, labels)
    train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
    
    # Initialize and train the GNN model
    print("Training the GNN model...")
    model = GraphSAGENet(input_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluate on test data
    print("Evaluating the model...")
    test_fcgs = []
    for apk_path in test_apk_paths:
        fcg = segdroid.extract_fcg(apk_path)
        pruned_fcg = segdroid.prune_graph(fcg)
        test_fcgs.append(pruned_fcg)
    
    test_data_list = segdroid.prepare_data_for_gnn(test_fcgs, test_labels)
    test_loader = DataLoader(test_data_list, batch_size=32)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch)
            pred = output.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Example of model explanation for the first test sample
    if test_data_list:
        print("Generating model explanation for a sample...")
        sample = test_data_list[0]
        node_importance = model.get_importance(sample)
        print(f"Node importance scores: Min={node_importance.min():.4f}, Max={node_importance.max():.4f}")
        
    return segdroid, model, accuracy

# Example usage:
train_apk_paths = ["mal1", "ben1.apk", "mal2", "ben2.apk", "mal3", "ben3.apk"]  # List of APK paths
train_labels = [0, 1, 0, 1, 0, 1]  # 0 for benign, 1 for malware
test_apk_paths = ["mal4", "mal5", "ben4.apk", "ben5.apk"] # List of APK paths
test_labels = [0, 0, 1, 1]  # 0 for benign, 1 for malware
segdroid, model, accuracy = train_segdroid(train_apk_paths, train_labels, test_apk_paths, test_labels)