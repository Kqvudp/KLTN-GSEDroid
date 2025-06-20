import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
cic_values = [0.9128, 0.9292, 0.9808, 0.9543]
drebin_values = [0.7917, 0.8105, 0.9671, 0.8819]

x = np.arange(len(metrics))  # Vị trí các nhóm
width = 0.35  # Độ rộng của các cột

fig, ax = plt.subplots(figsize=(10, 6))

# Vẽ cột cho CIC và DREBIN
rects1 = ax.bar(x - width/2, cic_values, width, label='CIC', color='#1f77b4')
rects2 = ax.bar(x + width/2, drebin_values, width, label='DREBIN', color='#ff7f0e')

# Thêm nhãn, tiêu đề và chú thích
ax.set_ylabel('Scores', fontsize=12)
ax.set_title('So sánh hiệu suất giữa CIC và DREBIN', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=12)

# Hiển thị giá trị trên mỗi cột
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

# Tùy chỉnh thêm
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()