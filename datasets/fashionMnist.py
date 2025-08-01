import torch
import torchvision
import torchvision.transforms as transforms

def get_loaders(batch_size):
    train_dataset = torchvision.datasets.FashionMNIST(root = './data',
                                               train = True,
                                               transform = transforms.Compose([
                                                      transforms.Resize((32,32)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean = (0.1307,), std = (0.3081,))
                                                      ]),
                                               download = True)
    
    
    test_dataset = torchvision.datasets.FashionMNIST(root = './data',
                                                train = False,
                                                transform = transforms.Compose([
                                                        transforms.Resize((32,32)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean = (0.1325,), std = (0.3105,))
                                                        ]),
                                                download=True)
        
        
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
        
        
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False)
    
    return train_loader, test_loader