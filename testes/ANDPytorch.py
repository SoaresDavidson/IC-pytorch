import torch
import matplotlib.pyplot as plt
import numpy as np
device = "cuda"

inputs = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32).to(device)
labels = torch.tensor([[0],[0],[0],[1]], dtype=torch.float32).to(device)

# print(inputs,labels)

num_epochs = 2000
inputDim = 2
outputDim = 1
lr = 0.1

class LinearRegression(torch.nn.Module):
    def __init__(self, inputDim, outputDim):
        super().__init__()
        self.linear = torch.nn.Linear(inputDim, outputDim)
    
    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

model = LinearRegression(inputDim, outputDim).to(device)

criterion = torch.nn.BCELoss()
optim = torch.optim.SGD(model.parameters(), lr=lr)

for i in range(num_epochs):
    input = inputs
    target = labels

    optim.zero_grad()

    output = model(input)
    
    loss = criterion(output, target)
    
    loss.backward()

    optim.step()
    print(loss)

predicted = model(inputs)
print(predicted)
    