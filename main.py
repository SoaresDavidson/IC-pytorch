import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
import models

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
# Define relevant variables for the ML task
batch_size = config["hyperparameters"]["batch_size"]
num_classes = config["hyperparameters"]["num_classes"]
learning_rate = config["hyperparameters"]["lr"]
num_epochs = config["hyperparameters"]["num_epochs"]


# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dataset_name = config['dataset']['name']
data_path = config['dataset']['path']

print(f'Carregando {dataset_name}')

train_loader, test_loader = load_dataset(name=dataset_name ,batch_size=batch_size)
    
model_name = config['model']['name']
model = models.get_model(model_name)(num_classes).to(device)
print(model.modules)

cost = nn.CrossEntropyLoss()
    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader.dataset) #type: ignore
print(total_step)
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        
        loss = cost(outputs, labels)
        
        loss.backward()

        optimizer.step()
        if (i) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i * len(images)}/{total_step}], Loss: {loss.item():.4f}')


model.eval() 

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images,labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
