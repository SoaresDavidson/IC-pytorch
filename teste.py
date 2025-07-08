# arquivo inutil apenas para testar ideias idiotas
import torch
import torch.nn as nn
##como funciona o keepdim=True
# x = torch.arange(2*2*2*2).resize_(2,2,2,2)
# print(x)
# y = x.sum((0,1,2),keepdim=True)
# print(y)
# print(y.size())

##teste de elemete-wise operations
# x = torch.arange(2*2*2*2).resize_(2,2,2,2)
# print((x-2)**2)

# x = torch.randn(10,84)
# print(x)

## KMean Cluster
import torch

def kmeans(X, k, num_iters=100, tol=1e-4):
    # Inicializa k centróides aleatórios a partir de pontos do dataset
    indices = torch.randperm(X.size(0))[:k]
    centroids = X[indices]
    # print(centroids)

    for i in range(num_iters):
        # Calcula distâncias euclidianas entre cada ponto e cada centróide (forma vetorizada)
        distances = torch.cdist(X, centroids)  # shape: (N, K)
        # print(distances)
        # Atribui cada ponto ao cluster mais próximo
        labels = distances.argmin(dim=1)
        # print(labels)
        # Recalcula centróides com a média dos pontos em cada cluster
        new_centroids = torch.stack([
            X[labels == j].mean(dim=0) if (labels == j).any() else centroids[j]
            for j in range(k)
        ])
        # print(new_centroids)

        # Verifica convergência
        if torch.norm(centroids - new_centroids) < tol:
            break

        centroids = new_centroids

    return labels, centroids

# Exemplo: dados com 84 amostras e 10 features
x = torch.randn(128,84).cuda()  # ou .to(device) se quiser generalizar
labels, centers = kmeans(x, k=3)

# print("Labels:", labels)
# print("Centers shape:", centers)

class RBF(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        _,self.centers = kmeans(input, k=10)
        
    def forward(self, input):

        diff = input.unsqueeze(1) - centers.unsqueeze(0) #[128,1,84] - [1, 10, 84]
        # diff shape: [128, 10, 84]
        result = 0
        norm = torch.linalg.norm(diff, ord=2, dim=2)  # shape: [128, 10]
        print(norm.shape)
        print(norm[:,1])
        print(self.weight.shape)
        result += self.weight * torch.exp(norm)[:,1]
        print(result.shape)
        return result

net = RBF(84, 1).cuda()
print(net(x))

