import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Normalização estilo LeNet-5
custom_normalization = transforms.Compose([
    transforms.ToTensor(),                        # [0,1]
    transforms.Lambda(lambda x: 1.175 * (1 - x))  # fundo branco → 0, traço preto → 1.175
])

# Dataset MNIST
original_transform = transforms.ToTensor()

mnist_raw = torchvision.datasets.MNIST(root='./data', train=True, transform=original_transform, download=True)
mnist_custom = torchvision.datasets.MNIST(root='./data', train=True, transform=custom_normalization, download=True)

# Pega a mesma imagem nos dois formatos
img_original, label = mnist_raw[0]
img_custom, _ = mnist_custom[0]

# Converte para 2D para visualização
img_original_2d = img_original.squeeze()
img_custom_2d = img_custom.squeeze()

# Visualização lado a lado
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

axs[0].imshow(img_original_2d, cmap='gray')
axs[0].set_title("Original MNIST (ToTensor)")
axs[0].axis('off')

axs[1].imshow(img_custom_2d, cmap='gray')
axs[1].set_title("LeNet-style Normalized")
axs[1].axis('off')

plt.tight_layout()
plt.show()
