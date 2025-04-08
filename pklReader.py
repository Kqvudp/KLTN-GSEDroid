import pickle

# Đọc file pickle
with open('processed_2c0c697f1adb39d2121ca123bdcbe82824be7f0618ebf551a5e514eedbb2f76b.pkl', 'rb') as file:  # 'rb' là chế độ đọc nhị phân
    data = pickle.load(file)
    
print(data)