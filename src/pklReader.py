import pickle

# Đọc file pickle
with open(r'D:\GraduateDissertation\Embedded\CIC\test\processed_0000764713b286cfe7e8e76c7038c92312977712d9c5a86d504be54f3c1d025a.c7a616849372467d03e51022a31b857a.pkl', 'rb') as file:  # 'rb' là chế độ đọc nhị phân
    data = pickle.load(file)
    
print(data)