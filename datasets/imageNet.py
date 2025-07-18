import torch
import torchvision
import torchvision.transforms as transforms

def get_loaders(batch_size):
    train_dataset = torchvision.datasets.ImageNet(root = './data',
                                               train = True,
                                               transform = transforms.Compose([
                                                        transforms.RandomResizedCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])
                                                        ]),
                                                        download = True)
    
    
    test_dataset = torchvision.datasets.ImageNet(root = './data',
                                                train = False,
                                                transform = transforms.Compose([
                                                        transforms.Resize(256),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])
                                                    ]),
                                                download=True)
        
        
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
        
        
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False)
    
    return train_loader, test_loader