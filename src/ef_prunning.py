from androguard.misc import AnalyzeAPK
import networkx as nx
import torch
import json
import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt

class FeatureExtractor:
    def __init__(self):
        self.permission_list = self._get_top_permissions()
        self.sensitive_apis = self._get_sensitive_apis()

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
    
    def save_graph_info_to_csv(self, file_name, original_graph, pruned_graph, csv_path):
        """Save graph information to a single CSV file"""
        original_nodes = len(original_graph.nodes)
        original_edges = len(original_graph.edges)
        pruned_nodes = len(pruned_graph.nodes)
        pruned_edges = len(pruned_graph.edges)
        reduction_nodes = original_nodes - pruned_nodes
        reduction_edges = original_edges - pruned_edges
        
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = ['File name', 'Original_node', 'Original_edge', 'Pruning_node', 'Pruning_edge', 'Reduction_node', 'Reduction_edge']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'File name': file_name,
                'Original_node': original_nodes,
                'Original_edge': original_edges,
                'Pruning_node': pruned_nodes,
                'Pruning_edge': pruned_edges,
                'Reduction_node': reduction_nodes,
                'Reduction_edge': reduction_edges
            })
    
    def _get_sensitive_apis(self):
        """Load sensitive API list from file"""
        sensitive_apis = set()
        api_file_path = "sensitive_apis.txt"
        
        try:
            with open(api_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    sensitive_apis.add(line)
            print(f"Loaded {len(sensitive_apis)} sensitive APIs from file")
        except Exception as e:
            print(f"Error loading sensitive APIs file: {str(e)}")

        return sensitive_apis

    def extract_apk_features(self, apk_path, file_name, csv_path):
        """Extract all features from an APK file"""
        print(f"Extracting features from: {apk_path}")
        a, d, dx = AnalyzeAPK(apk_path)
        
        features = {
            'permissions': self.extract_permissions(a),
            'call_graph': self.extract_call_graph(dx, file_name, csv_path),
        }
        
        return features

    def extract_permissions(self, apk_obj):
        """Extract permission features"""
        perm_vector = torch.zeros(len(self.permission_list))
        apk_perms = apk_obj.get_permissions()
        
        for i, perm in enumerate(self.permission_list):
            if perm in apk_perms:
                perm_vector[i] = 1
                
        return perm_vector.tolist()

    def extract_call_graph(self, analysis_obj, file_name, csv_path):
        """Extract call graph features"""
        graph = nx.DiGraph()
        
        # Build the initial graph
        for method in analysis_obj.get_methods():

            if method.is_external():
                continue
            method_str = str(method)
            
            try:
                opcodes = [inst.get_name() for inst in method.get_method().get_instructions()]
            except AttributeError:
                opcodes = []
            
            graph.add_node(method_str, opcodes=opcodes)
            
            for called in method.get_xref_to():
                called_method = str(called[1])
                graph.add_edge(method_str, called_method)

        original_nodes = len(graph.nodes)
        original_edges = len(graph.edges)
        print(f"Original FCG: {original_nodes} nodes, {original_edges} edges")
        
        threshold = 100
        pruned_graph = self.prune_graph(graph, threshold)
        
        pruned_nodes = len(pruned_graph.nodes)
        pruned_edges = len(pruned_graph.edges)
        print(f"Pruned FCG: {pruned_nodes} nodes, {pruned_edges} edges")
        print(f"Reduction: {original_nodes - pruned_nodes} nodes ({(original_nodes - pruned_nodes)/original_nodes*100:.2f}%), "
              f"{original_edges - pruned_edges} edges ({(original_edges - pruned_edges)/original_edges*100:.2f}%)")
        self.save_graph_info_to_csv(file_name, graph, pruned_graph, csv_path)

        return {
            'nodes': list(pruned_graph.nodes(data=True)),
            'edges': list(pruned_graph.edges()),
        }
        
    def prune_graph(self, fcg, threshold=100):
        """Prune FCG using sensitive API-based approach"""
        # Skip pruning for small graphs
        if len(fcg.nodes) <= threshold:
            print(f"Graph size ({len(fcg.nodes)}) below threshold ({threshold}), skipping pruning")
            return fcg
            
        # Find sensitive nodes
        sensitive_nodes = set()
        sensitive_api_counts = {}
        
        for node, attrs in fcg.nodes(data=True):
            if attrs.get('is_sensitive', False):
                sensitive_nodes.add(node)
                # Check outgoing edges to find which sensitive APIs are called
                for _, called_method in fcg.out_edges(node):
                    # Match against sensitive APIs
                    for api in self.sensitive_apis:
                        if api == called_method or api in called_method:
                            sensitive_api_counts[api] = sensitive_api_counts.get(api, 0) + 1
                            break
                # After the outgoing edges loop
                if node in sensitive_nodes and not any(api_count for _, api_count in sensitive_api_counts.items()):
                    print(f"Warning: Node marked sensitive but no API match found: {node[:100]}")
    
        if not sensitive_nodes:
            print("No sensitive nodes found, returning original graph")
            return fcg
        
        print(f"Found {len(sensitive_nodes)} sensitive nodes")

        # Find ancestor nodes (upward direction)
        ancestor_nodes = set()
        for sv in sensitive_nodes:
            # Use depth-first search to find ancestors
            try:
                ancestors = nx.ancestors(fcg, sv)
                ancestor_nodes.update(ancestors)
                ancestor_nodes.add(sv)
            except nx.NetworkXError:
                # Handle case where node might not exist in the graph
                pass
                
        print(f"Found {len(ancestor_nodes)} ancestor nodes")
                
        # Find descendant nodes (downward direction)
        descendant_nodes = set()
        for sav in ancestor_nodes:
            # Use depth-first search to find descendants
            try:
                descendants = nx.descendants(fcg, sav)
                descendant_nodes.update(descendants)
                descendant_nodes.add(sav)
            except nx.NetworkXError:
                # Handle case where node might not exist in the graph
                pass
                
        print(f"Found {len(descendant_nodes)} descendant nodes")
                
        # Create the pruned graph
        all_preserved_nodes = ancestor_nodes.union(descendant_nodes)
        print(f"Total preserved nodes: {len(all_preserved_nodes)}")
        
        pruned_fcg = fcg.subgraph(all_preserved_nodes).copy()
        
        return pruned_fcg

def save_features(features, output_path):
    """Save extracted features to a JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"Saved features to {output_path}")

def process_apk_folder(input_folder, output_folder, label=0):
    """Process all APKs in a folder and extract their features"""
    extractor = FeatureExtractor()
    
    for apk_file in os.listdir(input_folder):

        apk_path = os.path.join(input_folder, apk_file)
        output_path = os.path.join(output_folder, f"{apk_file}.json")
        csv_path = f"graph_info.csv"
        
        try:
            features = extractor.extract_apk_features(apk_path, apk_file, csv_path)
            features['label'] = label
            save_features(features, output_path)
            
        except Exception as e:
            print(f"Error processing {apk_file}: {str(e)}")

if __name__ == "__main__":
    # malware_folder = r"E:\Raw\CIC\Adware"
    # output_malware_folder = r"E:\Extracted\CIC\Adware\After_Prunning"
    # os.makedirs(output_malware_folder, exist_ok=True)

    # malware_folder = r"E:\Raw\CIC\Riskware"
    # output_malware_folder = r"E:\Extracted\CIC\Riskware\After_Prunning"
    # os.makedirs(output_malware_folder, exist_ok=True)

    malware_folder = r"E:\Raw\CIC\Banking"
    output_malware_folder = r"E:\Extracted\CIC\Banking\After_Prunning"
    os.makedirs(output_malware_folder, exist_ok=True)

    # malware_folder = r"E:\Raw\CIC\SMS"
    # output_malware_folder = r"E:\Extracted\CIC\SMS\After_Prunning"
    # os.makedirs(output_malware_folder, exist_ok=True)

    # malware_folder = r"E:\Raw\Drebin"
    # output_malware_folder = r"E:\Extracted\Drebin\After_Prunning"
    # os.makedirs(output_malware_folder, exist_ok=True)
    
    print("Processing malware APKs...")
    process_apk_folder(malware_folder, output_malware_folder, label=1)
