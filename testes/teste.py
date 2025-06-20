import torch
import numpy as np
import matplotlib.pyplot as plt
# x = torch.randn(4,3)
# print(x.size())
# print(x.reshape(-1,2,2,1))
# print(x)

# x = torch.randn(3,4)
# y = torch.randn(4,3)
# print(x)
# print(y)
# y = torch.transpose(y,0 ,1)
# print(x + x)

"concatenar por eixo"
# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))

"broadcasting mechanism"
# a = torch.arange(12).reshape((3, 4))
# b = torch.arange(4).reshape((1, 4))
# print(a, b)
# print(a+b)

# A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
# B = A.clone()  # Assign a copy of A to B by allocating new memory
# print(A, A + B)
# print(id(A) == id(B))

# x = torch.arange(12).reshape(3, 2, 2)
# print(x)
# print(x.sum(axis=0))
# print(x.sum(axis=1))

# x = torch.rand(2,3)
# print(x)
# print(x/x.sum(axis=0))

# x = torch.arange(4.0)
# # Can also create x = torch.arange(4.0, requires_grad=True)
# x.requires_grad_(True)
# print(x.grad)  # The gradient is None by default
# y = 2 * torch.dot(x, x)
# print(y)

# print(y.backward())
# print(x.grad)

# x.grad.zero_()  # Reset the gradient
# y = x.sum()
# y.backward()
# print(x.grad)

x = torch.arange(24,dtype=float).reshape(3,2,2,2)
print(x)
print(x.mean(dim=1))


# inputDim = 1        # takes variable 'x' 
# outputDim = 1       # takes variable 'y'
# learningRate = 0.01 
# epochs = 100
# device = "cuda"
# # create dummy data for training
# x_values = [i for i in range(11)]
# x_train = np.array(x_values, dtype=np.float32)
# # print(x_train)
# x_train = x_train.reshape(-1, 1)
# # print(x_train)

# y_values = [2*i + 1 for i in x_values]
# y_train = np.array(y_values, dtype=np.float32)
# y_train = y_train.reshape(-1, 1)

# class linearRegression(torch.nn.Module):
#     def __init__(self, inputSize, outputSize):
#         super().__init__()
#         self.linear = torch.nn.Linear(inputSize, outputSize)

#     def forward(self, x):
#         out = self.linear(x)
#         return out

# model = linearRegression(inputDim, outputDim).to(device=device)
# print(model.parameters())
# criterion = torch.nn.MSELoss() 
# optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

# for epoch in range(epochs):
#     # Converting inputs and labels to Variable
#     inputs = torch.from_numpy(x_train).cuda()
#     labels = torch.from_numpy(y_train).cuda()

#     # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
#     optimizer.zero_grad()

#     # get output from the model, given the inputs
#     outputs = model(inputs)

#     # get loss for the predicted output
#     loss = criterion(outputs, labels)
#     print(loss)
#     # get gradients w.r.t to parameters
#     loss.backward()

#     # update parameters
#     optimizer.step()

#     print('epoch {}, loss {}'.format(epoch, loss.item()))


# with torch.no_grad(): # we don't need gradients in the testing phase
#     predicted = model(torch.from_numpy(x_train).to(device=device))
#     print(predicted)

# plt.clf()
# plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
# plt.plot(x_train, predicted.cpu(), '--', label='Predictions', alpha=0.5)
# plt.legend(loc='best')
# plt.show()
