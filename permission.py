# feature_extractor.py
import csv
from androguard.core.bytecodes import apk, dvm
from androguard.core.analysis import analysis
import networkx as nx
import numpy as np

class FeatureExtractor:
    def __init__(self, apk_path):
        self.apk_path = apk_path
        self.a = None
        self.d = None
        self.dx = None
        
    def load_apk(self):
        """Load and parse APK file"""
        try:
            self.a = apk.APK(self.apk_path)
            self.d = dvm.DalvikVMFormat(self.a.get_dex())
            self.dx = analysis.Analysis(self.d)
            self.d.set_vmanalysis(self.dx)
            return True
        except Exception as e:
            print(f"Error loading APK: {e}")
            return False
            
    def extract_permissions(self):
        """Extract permission features"""
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
        
        permission_vector = np.zeros(len(all_permissions))
        apk_permissions = self.a.get_permissions()
        
        for i, permission in enumerate(all_permissions):
            if permission in apk_permissions:
                permission_vector[i] = 1
                
        return permission_vector
        
    def extract_api_calls(self):
        """Extract API call graph"""
        graph = nx.DiGraph()
        
        for method in self.d.get_methods():
            method_name = method.get_name()
            graph.add_node(method_name)
            
            if method.get_code():
                for instruction in method.get_code().get_bc().get_instructions():
                    if instruction.get_name().startswith('invoke-'):
                        called_method = instruction.get_output().split(',')[-1].strip()
                        graph.add_edge(method_name, called_method)
        return graph
        
    def extract_opcodes(self):
        """Extract opcode sequences"""
        opcode_sequences = []
        
        for method in self.d.get_methods():
            if method.get_code():
                sequence = []
                for instruction in method.get_code().get_bc().get_instructions():
                    sequence.append(instruction.get_name())
                if sequence:
                    opcode_sequences.append(sequence)
                    
        return opcode_sequences

def main():
    extractor = FeatureExtractor("sample")
    if extractor.load_apk():
        # Extract features
        permissions = extractor.extract_permissions()
        api_graph = extractor.extract_api_calls()
        opcodes = extractor.extract_opcodes()

        print(f"Extracted {len(permissions)} permissions")
        print(f"API call graph has {len(api_graph.nodes)} nodes and {len(api_graph.edges)} edges")
        print(f"Extracted {len(opcodes)} opcode sequences")

if __name__ == "__main__":
    main()