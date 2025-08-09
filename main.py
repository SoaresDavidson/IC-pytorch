import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import yaml
from datasets import load_dataset
import time
import models
from torchvision.utils import make_grid 
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models_t
from torchinfo import summary



with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    


batch_size = config["hyperparameters"]["batch_size"]
num_classes = config["hyperparameters"]["num_classes"]
learning_rate = config["hyperparameters"]["lr"]
num_epochs = config["hyperparameters"]["num_epochs"]

dataset_name = config['dataset']['name']
model_name = config['model']['name']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"dispositivo: {device}")


print(f'Carregando {dataset_name}')

train_loader, test_loader = load_dataset(name=dataset_name ,batch_size=batch_size)

    
model:nn.Module = models.get_model(model_name)(num_classes).to(device)
# model = models_t.resnet18()
i, _ = next(iter(train_loader))
summary(model, input_size=i.shape) #Ajuste o input para o seu modelo
print(model.modules)
# model.compile()
torch.set_float32_matmul_precision('high') 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4, foreach=True)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//4, gamma=0.1)
cost = nn.CrossEntropyLoss(label_smoothing=0.1)
# writer = SummaryWriter(f'logs/{dataset_name}')


total_samples = len(train_loader.dataset) #type: ignore
print(total_samples)
print(len(train_loader))

def train(epoch):
    model.train()
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = cost(outputs, labels)
        
        loss.backward()

        running_loss += loss.item()

        optimizer.step()

        if (i) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Sample [{i * batch_size}/{total_samples}], Loss: {loss.item():.4f}')
    # scheduler.step()

            # writer.add_scalar(f'training loss/{dataset_name}',
            #                     running_loss / 100,
            #                     epoch * len(train_loader) + i)

            #     # ...log a Matplotlib Figure showing the model's predictions on a
            #     # random mini-batch
            # writer.add_figure(f'predictions vs. actuals/{dataset_name}',
            #                 plot_classes_preds(outputs, images, labels),
            #                 global_step=epoch * len(train_loader) + i)
            # running_loss = 0.0

def eval():
    model.eval() 
    times = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images,labels = images.to(device), labels.to(device)

            start_time = time.time()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            end_time = time.time()
            times.append(end_time - start_time)

            total += labels.size(0)
            correct += (predicted == labels).sum().item() 

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

# def write_tensorBoard():
#     # get some random training images
#     dataiter = iter(train_loader)
#     images, labels = next(dataiter)
#     images = images.to(device)
#     # create grid of images
#     img_grid = make_grid(images)

#     # show images
#     matplotlib_imshow(img_grid, one_channel=True)

#     # write to tensorboard
#     writer.add_image(dataset_name, img_grid)
    
#     writer.add_graph(model, images)
#     writer.close()

# write_tensorBoard()

for i in range(num_epochs):
    train(i)
    eval()

    
# model.cpu()
# export_onnx_qcdq(model, input_shape=(1, 1, 32, 32), export_path="quant_model2.onnx" )

# traced = torch.fx.symbolic_trace(model).print_readable()


# dummy_input = torch.randn(1, 1, 32, 32).to(device)
# traced_model = torch.jit.trace(model, dummy_input)
# torch.jit.save(traced_model,"teste.pt")