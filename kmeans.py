import torch
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

x = torch.randn(84, 10)  
x_numpy = x.numpy()    

km = KMeans(n_clusters=10)
labels = km.fit_predict(x_numpy) 
print("Labels:", labels)
print("Centers:", km.cluster_centers_)

# plt.scatter(x_numpy[:, 0], x_numpy[:, 1], c=labels, cmap='viridis', label='Points')
# plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
#             c='red', marker='X', s=200, label='Centers')
# plt.title("KMeans Clusters")
# plt.legend()
# plt.show()
