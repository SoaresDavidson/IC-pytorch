import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os

class UTKFaceDataset(Dataset):
  
    def __init__(self, paths_and_labels, transform=None):
        self.paths_and_labels = paths_and_labels
        self.transform = transform

    def __len__(self):
        return len(self.paths_and_labels)

    def __getitem__(self, idx):

        img_path, label = self.paths_and_labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_loaders(batch_size):
    base_dir = os.getcwd() +'/data/jangedoo'
    print(base_dir)
    dataset_path = os.path.join(base_dir, "utkface-new","versions/1", "utkface_aligned_cropped", "UTKFace")

    image_files = os.listdir(dataset_path)
    image_paths_and_labels = []

    for filename in image_files:
        if not filename.endswith(".jpg"):
            continue
        try:
            parts = filename.split('_')
            gender_label = int(parts[1])
            full_path = os.path.join(dataset_path, filename)
            image_paths_and_labels.append((full_path, gender_label))
        except (IndexError, ValueError):
            print(f"Ignorando arquivo com nome mal formatado: {filename}")


    print(f"Encontradas {len(image_paths_and_labels)} imagens.")

    data_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    meu_dataset_genero = UTKFaceDataset(
        paths_and_labels=image_paths_and_labels, 
        transform=data_transforms
    )

    train_size = int(0.8 * len(meu_dataset_genero))
    val_size = len(meu_dataset_genero) - train_size
    train_ds, val_ds = random_split(meu_dataset_genero, [train_size, val_size])


    train_loader = DataLoader(
        dataset=train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=16
    )
    test_loader = DataLoader(
        dataset=val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=16
    )
    
    return train_loader, test_loader