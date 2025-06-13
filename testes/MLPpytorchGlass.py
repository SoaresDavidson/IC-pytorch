from collections import OrderedDict
from ucimlrepo import fetch_ucirepo
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

glass_identification = fetch_ucirepo(id=42)
inputs = glass_identification.data.features
targets = glass_identification.data.targets

#print(glass_identification.variables)

inputsTensor = torch.tensor(inputs.values, dtype=torch.float32)
targetsTensor = (torch.tensor(targets.values, dtype=torch.int64) - 1).squeeze() #evitar problema resultante do tamanho do tensor
scaler = StandardScaler()
inputsNormalized = scaler.fit_transform(inputsTensor)
inputsTensor = torch.tensor(inputsNormalized, dtype=torch.float32)

insTrain, insTest, outsTrain, outsTest = train_test_split(inputsTensor, targetsTensor, test_size=0.1, random_state=42)

model = nn.Sequential(OrderedDict([
    ('inputLayer', nn.Linear(9, 16)),
    ('relu1', nn.ReLU()),
    ('hiddenLayer', nn.Linear(16, 32)),
    ('relu2', nn.ReLU()),
    ('outputLayer', nn.Linear(32, 7))
]))

lossCalc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 1000

def trainModel(model, insTrain, outsTrain, lossCalc, optimizer):
    for epoch in range(epochs):
        model.train()
        outputs = model(insTrain)
        loss = lossCalc(outputs, outsTrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

trainModel(model, insTrain, outsTrain, lossCalc, optimizer)

model.eval()
with torch.no_grad():
    outsPred = model(insTest)
    _, predicted_classes = torch.max(outsPred, 1)
    accuracy = accuracy_score(outsTest.numpy(), predicted_classes.numpy())
print(f'Acurácia no teste: {accuracy * 100:.2f}%')

"""
Observações:
    Problema no carregamento dos tensores de alvos. 
    Cálculo da perda e tipos específicos de variáveis.
    Adição de otimizadores e normalizadores.
    Aumento do número de épocas: 100 -> 1000 (~50% -> ~80% de acurácia).
    Aumentos consecutivos não resultaram numa melhoria significativa.
    Adição de uma camada oculta (~80% -> 85% de acurácia)
"""