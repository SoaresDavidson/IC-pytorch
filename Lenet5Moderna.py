import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml
from datasets import load_dataset

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
# Define relevant variables for the ML task
batch_size = config["hyperparameters"]["batch_size"]
num_classes = config["hyperparameters"]["num_classes"]
learning_rate = config["hyperparameters"]["lr"]
num_epochs = config["hyperparameters"]["num_epochs"]
    
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = config['dataset']['name']
data_path = config['dataset']['path']

print(f'Carregando {dataset_name}')

train_loader, test_loader = load_dataset(name=dataset_name ,batch_size=batch_size)

#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.linearLayer = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linearLayer(out)
        return out
    
model = LeNet5(num_classes).to(device)
    

cost = nn.CrossEntropyLoss()
    

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    

total_step = len(train_loader)

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)
            

        outputs = model(images)
        loss = cost(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 400 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))


model.eval() 

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
