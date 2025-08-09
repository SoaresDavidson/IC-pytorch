import torch
import torchvision
import torchvision.transforms as t
import os

def get_loaders(batch_size):
    transform = t.Compose([
            t.Resize((256, 256)),
            t.CenterCrop(224),
            t.ToTensor(),
            t.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            ])
    
    caminho = os.getcwd()
    treino = "/data/potholes/Train data"
    teste = "/data/potholes/Test data"

    dataset_treino = torchvision.datasets.ImageFolder(caminho+ treino, transform=transform)
    dataset_validacao = torchvision.datasets.ImageFolder(caminho+teste, transform=transform)

        
    train_loader = torch.utils.data.DataLoader(dataset = dataset_treino,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                pin_memory=True,
                                                num_workers=16
                                                )
        
        
    test_loader = torch.utils.data.DataLoader(dataset = dataset_validacao,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                pin_memory=True,
                                                num_workers=16
                                                )
    
    return train_loader, test_loader