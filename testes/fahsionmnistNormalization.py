import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# Load one raw image (as PIL)
raw_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
image, label = raw_dataset[0]  # Get first image and label
print(f"Label: {label}")
print(f"Original type: {type(image)}, size: {image.size}")

# Step 1: Resize
resized_image = transforms.Resize((32, 32))(image)
print(f"Resized size: {resized_image.size}")

# Step 2: ToTensor
tensor_image = transforms.ToTensor()(resized_image)

# Step 3: Normalize
normalized_image = transforms.Normalize(mean=(0.1307,), std=(0.3081,))(tensor_image)

# Plot all stages
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(image, cmap='gray')
axs[0].set_title("Original (28x28)")
axs[1].imshow(resized_image, cmap='gray')
axs[1].set_title("Resized (32x32)")
axs[2].imshow(tensor_image.squeeze(), cmap='gray')
axs[2].set_title("ToTensor (0–1)")
axs[3].imshow(normalized_image.squeeze(), cmap='gray')
axs[3].set_title("Normalized")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
