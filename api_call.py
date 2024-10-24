from androguard.core.bytecodes.apk import APK
from androguard.core.analysis.analysis import Analysis
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.misc import AnalyzeAPK

class APIExtractor:
    def __init__(self, d):
        self.d = d

    def extract_api_calls(self):
        """Extract API calls"""
        api_calls = []
        methods_calling_apis = set()
        methods_called_by_apis = set()
        third_party_prefixes = [
            'Lorg/codehaus/', 'Lcom/apperhand/', 'Lcom/airpush/', 'Lcom/mobclix/'
        ]

        for method in self.d.get_methods():
            method_name = method.get_name()

            if method.get_code():
                for instruction in method.get_code().get_bc().get_instructions():
                    if instruction.get_name().startswith('invoke-'):
                        called_method = instruction.get_output().split(',')[-1].strip()
                        if not any(called_method.startswith(prefix) for prefix in third_party_prefixes):
                            methods_calling_apis.add(method_name)
                            methods_called_by_apis.add(called_method)
                            api_calls.append(called_method)

        # Filter out isolated nodes and format API calls
        filtered_api_calls = [
            # call.lstrip('L').split('(')[0] for call in set(api_calls)  # Use set to remove duplicates
            call for call in set(api_calls)  # Use set to remove duplicates
            if call in methods_calling_apis or call in methods_called_by_apis
        ]

        return filtered_api_calls
    
# Load APK and extract information
apk_path = "sample"  # Update with the actual APK path
apk, dex_list, dx = AnalyzeAPK(apk_path)

total_api_calls = 0

for dex in dex_list:
    extractor = APIExtractor(dex)  # Initialize APIExtractor for each DEX
    api_calls = extractor.extract_api_calls()  # Extract API calls
    total_api_calls += len(api_calls)


for call in api_calls:
    print(call)


print(f"Total API calls: {total_api_calls}")