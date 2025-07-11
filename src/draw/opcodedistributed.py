import matplotlib.pyplot as plt
import numpy as np

# Tạo dữ liệu chính (phân bố bình thường)
np.random.seed(42)
normal_data = np.random.normal(loc=100, scale=30, size=200)

# Tăng mật độ mũi tên trong khoảng 250–300
dense_outliers = np.random.uniform(low=250, high=300, size=15)

# Các outliers khác rải rác
other_outliers = np.random.uniform(low=301, high=500, size=8)

# Tổng hợp dữ liệu
data = np.concatenate((normal_data, dense_outliers, other_outliers))

# Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(6, 8))

box = ax.boxplot(data,
                 patch_artist=True,
                 boxprops=dict(facecolor='lightblue'),
                 flierprops=dict(marker='^',
                                 markerfacecolor='orange',
                                 markeredgecolor='orange',
                                 markersize=10),
                 medianprops=dict(color='red'))

# Gán nhãn và tiêu đề
ax.set_ylabel('Opcode count')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
