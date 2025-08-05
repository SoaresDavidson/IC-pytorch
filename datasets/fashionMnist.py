import torch
import torchvision
import torchvision.transforms as transforms

def get_loaders(batch_size):
    transform =  transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = (0.1307,), std = (0.3081,))
                                ])
    
    train_dataset = torchvision.datasets.FashionMNIST(root = './data',
                                               train = True,
                                               transform = transform,
                                               download = True)
    
    
    test_dataset = torchvision.datasets.FashionMNIST(root = './data',
                                                train = False,
                                                transform = transform,
                                                download=True)
        
        
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                pin_memory=True,
                                                num_workers=16
                                                )
        
        
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                pin_memory=True,
                                                num_workers=16
                                                )
    
    return train_loader, test_loader