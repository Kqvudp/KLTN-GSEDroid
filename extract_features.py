from androguard.misc import AnalyzeAPK
import networkx as nx
import torch
import json
import os
from pathlib import Path

class FeatureExtractor:
    def __init__(self):
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

    

    def extract_apk_features(self, apk_path):
        """Extract all features from an APK file"""
        print(f"Extracting features from: {apk_path}")
        a, d, dx = AnalyzeAPK(apk_path)
        
        features = {
            'permissions': self.extract_permissions(a),
            'call_graph': self.extract_call_graph(dx),
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

    def extract_call_graph(self, analysis_obj):
        """Extract call graph features"""
        graph = nx.DiGraph()
        
        for method in analysis_obj.get_methods():
            if method.is_external():
                continue
            
            has_edges = False
            opcodes = [inst.get_name() for inst in method.get_method().get_instructions()]
            for called in method.get_xref_to():
                if not called[1].is_external():
                    has_edges = True
                    graph.add_edge(str(method), str(called[1]))
            
            if has_edges:
                graph.add_node(str(method), opcodes=opcodes)

        return {
            'nodes': list(graph.nodes(data=True)),
            'edges': list(graph.edges())
        }

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
        
        try:
            features = extractor.extract_apk_features(apk_path)
            features['label'] = label
            save_features(features, output_path)
            
        except Exception as e:
            print(f"Error processing {apk_file}: {str(e)}")

if __name__ == "__main__":
    # benign_folder = r"D:\FinalProject\input\bengin"
    malware_folder = r"D:\FinalProject\code\main_v6\malware"
    output_folder = r"D:\FinalProject\code\main_v6\extract"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # print("Processing benign APKs...")
    # process_apk_folder(benign_folder, output_folder, label=0)
    
    print("Processing malware APKs...")
    process_apk_folder(malware_folder, output_folder, label=1)