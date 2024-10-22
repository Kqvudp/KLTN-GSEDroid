from androguard.core.bytecodes import dvm
from androguard.misc import AnalyzeDex

import xml.etree.ElementTree as ET

def extract_permissions_from_manifest(manifest_path):
    permissions = []

    # Parse the XML file
    tree = ET.parse(manifest_path)
    root = tree.getroot()

    # Namespace for AndroidManifest.xml
    ns = {'android': 'http://schemas.android.com/apk/res/android'}

    # Find all permissions
    for permission in root.findall('uses-permission', ns):
        permissions.append(permission.attrib['{http://schemas.android.com/apk/res/android}name'])

    return permissions

def convert_permissions_to_binary_vector(extracted_permissions, all_possible_permissions):
    binary_vector = []
    for permission in all_possible_permissions:
        if permission in extracted_permissions:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    return binary_vector

# Use the function with the path to your AndroidManifest.xml file
manifest_path = 'AndroidManifest.xml'  # Change the path
extracted_permissions = extract_permissions_from_manifest(manifest_path)

# Define all possible permissions (this is a simplified example, the actual list can be much longer)
all_possible_permissions = [
    'android.permission.INTERNET'
    'android.permission.ACCESS_NETWORK_STATE'
    'android.permission.WRITE_EXTERNAL_STORAGE'
    'android.permission.ACCESS_WIFI_STATE'
    'android.permission.READ_PHONE_STATE'
    'android.permission.WAKE_LOCK'
    'android.permission.VIBRATE'
    'android.permission.READ_EXTERNAL_STORAGE'
    'android.permission.ACCESS_FINE_LOCATION'
    'android.permission.ACCESS_COARSE_LOCATION'
    # Add more permissions as needed
]

# Convert extracted permissions to binary vector
binary_vector = convert_permissions_to_binary_vector(extracted_permissions, all_possible_permissions)

# Print the binary vector
print("Binary vector for permissions:", binary_vector)