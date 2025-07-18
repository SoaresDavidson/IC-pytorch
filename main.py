import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
import time
import models
from util import BinOp, plot_batch
from torchvision.models import alexnet


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
# Define relevant variables for the ML task
batch_size = config["hyperparameters"]["batch_size"]
num_classes = config["hyperparameters"]["num_classes"]
learning_rate = config["hyperparameters"]["lr"]
num_epochs = config["hyperparameters"]["num_epochs"]
bin = config["hyperparameters"]["binarize"]


# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dataset_name = config['dataset']['name']
data_path = config['dataset']['path']

print(f'Carregando {dataset_name}')

train_loader, test_loader = load_dataset(name=dataset_name ,batch_size=batch_size)
    
model_name = config['model']['name']
# model = models.get_model(model_name)(num_classes).to(device)
model = alexnet(num_classes=100).to(device)
print(model.modules)

cost = nn.CrossEntropyLoss()
    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
binop = BinOp(model=model)
total_samples = len(train_loader.dataset) #type: ignore
print(total_samples)
plot_batch(test_loader, test_loader.dataset.classes)
def train(epoch, bin=False):
    model.train()
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)

        if bin:
            binop.binarization()

        optimizer.zero_grad()

        outputs = model(images)
        
        loss = cost(outputs, labels)
        
        loss.backward()
        if bin:
            binop.restore()
            binop.updateBinaryGradWeight()

        optimizer.step()
        if (i) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Sample [{i * batch_size}/{total_samples}], Loss: {loss.item():.4f}')

def eval(bin=False):
    model.eval() 
    times = []
    with torch.no_grad():
        correct = 0
        total = 0
        if bin:
            binop.binarization()
        for images, labels in test_loader:
            images,labels = images.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            end_time = time.time()
            times.append(end_time - start_time)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
        # if bin:
            # binop.restore()

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
        avg_inference_time = sum(times) / len(times) if times else 0
        print(f"Average inference time per batch: {avg_inference_time:.6f} seconds")

for i in range(num_epochs):
    train(i,bin)
    eval(bin)