import torch
import torchvision
import torchvision.transforms as transforms

def get_loaders(batch_size):
    train_dataset = torchvision.datasets.CIFAR10(root = './data',
                                               train = True,
                                               transform = transforms.Compose([
                                                #       transforms.Resize((224, 224)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                                                      ]),
                                               download = True)
    
    
    test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                                train = False,
                                                transform = transforms.Compose([
                                                        # transforms.Resize((224, 224)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                                                        ]),
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