import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import yaml
from datasets import load_dataset
import time
import models
from util import BinOp, plot_classes_preds
from torchvision.utils import make_grid 
import matplotlib.pyplot as plt
import numpy as np


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    


batch_size = config["hyperparameters"]["batch_size"]
num_classes = config["hyperparameters"]["num_classes"]
learning_rate = config["hyperparameters"]["lr"]
num_epochs = config["hyperparameters"]["num_epochs"]
bin = config["hyperparameters"]["binarize"]
dataset_name = config['dataset']['name']
model_name = config['model']['name']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"dispositivo: {device}")


print(f'Carregando {dataset_name}')

train_loader, test_loader = load_dataset(name=dataset_name ,batch_size=batch_size)

    
model = models.get_model(model_name)(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
cost = nn.CrossEntropyLoss()
writer = SummaryWriter(f'runs/{dataset_name}')
binop = BinOp(model=model)


total_samples = len(train_loader.dataset) #type: ignore
# print(model.modules)
print(total_samples)
print(len(train_loader))

def train(epoch, bin=False):
    model.train()
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)

        if bin:
            binop.binarization()

        optimizer.zero_grad()

        outputs = model(images)
        # for param in model.parameters():
        #     print(param)
        #     print(param.shape)
        loss = cost(outputs, labels)
        
        loss.backward()

        running_loss += loss.item()
        if bin:
            binop.restore()
            binop.updateBinaryGradWeight()

        optimizer.step()
        if (i) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Sample [{i * batch_size}/{total_samples}], Loss: {loss.item():.4f}')

            writer.add_scalar(f'training loss/{dataset_name}',
                                running_loss / 100,
                                epoch * len(train_loader) + i)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
            writer.add_figure(f'predictions vs. actuals/{dataset_name}',
                            plot_classes_preds(outputs, images, labels),
                            global_step=epoch * len(train_loader) + i)
            running_loss = 0.0

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

        if bin:
            binop.restore()

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
        avg_inference_time = sum(times) / len(times) if times else 0
        print(f"Average inference time per batch: {avg_inference_time:.6f} seconds")

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def write_tensorBoard():
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    # create grid of images
    img_grid = make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image(dataset_name, img_grid)
    
    writer.add_graph(model, images)
    writer.close()

write_tensorBoard()

for i in range(num_epochs):
    train(i,bin)
    eval(bin)


if bin:
    binop.binarization()
torch.jit.trace(model, torch.randn(1, 1, 32, 32))
torch.jit.save(model.state_dict(),"teste.pt")