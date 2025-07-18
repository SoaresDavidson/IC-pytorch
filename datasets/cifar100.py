import torch
import torchvision
import torchvision.transforms as transforms

def get_loaders(batch_size):
    train_dataset = torchvision.datasets.CIFAR100(root = './data',
                                               train = True,
                                               download = True,
                                               transform = transforms.Compose([
                                                transforms.Resize(256),               # Resize to slightly larger
                                                transforms.RandomCrop(224),           # Random crop to 224×224
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
                                            ]   )
                                            )
    
    
    test_dataset = torchvision.datasets.CIFAR100(root = './data',
                                                train = False,
                                                download=True,
                                                transform = transforms.Compose([
                                                       transforms.Resize(224),               # Directly resize test set
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
                                                    ]),
                                                )
        
        
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
        
        
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False)
    
    return train_loader, test_loader
