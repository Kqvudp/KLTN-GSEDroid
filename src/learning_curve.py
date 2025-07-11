import networkx as nx
import matplotlib.pyplot as plt

# Tạo đồ thị ban đầu (trước khi cắt tỉa)
G_before = nx.Graph()
G_before.add_edges_from([
    (1, 2), (1, 3), (2, 4), (3, 4), (3, 5),
    (5, 6), (6, 7), (7, 8), (8, 9)
])

# Tạo đồ thị sau khi cắt tỉa (ví dụ: bỏ các node độ bậc < 2)
G_after = G_before.copy()
low_degree_nodes = [node for node, degree in dict(G_after.degree()).items() if degree < 2]
G_after.remove_nodes_from(low_degree_nodes)

# Thiết lập layout chung để 2 hình dễ so sánh
pos = nx.spring_layout(G_before, seed=42)

# Vẽ đồ thị trước khi cắt tỉa
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
nx.draw(G_before, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)
plt.title("Trước khi cắt tỉa")

# Vẽ đồ thị sau khi cắt tỉa
plt.subplot(1, 2, 2)
nx.draw(G_after, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=700)
plt.title("Sau khi cắt tỉa")

plt.tight_layout()
plt.show()
