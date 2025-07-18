import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class BinOp():
    def __init__(self, model): #pega as camadas que vão ter os pesos binarizados
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # print(m)
                count_targets = count_targets + 1

        start_range = 1
        end_range = count_targets-2
        # print(count_targets)
        self.bin_range = list(range(start_range, end_range + 1))
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []
        # print(self.bin_range)
        # print(self.num_of_params)
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                print(m)
                if index in self.bin_range: #checagem a mais?
                    #tmp = m.weight.data.clone()
                    with torch.no_grad():
                        tmp = m.weight.clone()
                    # print(m.weight)
                    # print(m.weight.size())
                    self.saved_params.append(tmp) #salva os pesos
                    self.target_modules.append(m.weight)
        return

    def binarization(self): #faz tudo
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self): #calcula a média absoluta dos pesos e subtrai dele mesmo (centraliza os pesos)
        for param in self.target_modules:
            # s = self.target_modules[index].data.size() não é utilizado
            with torch.no_grad():
                negMean = param.mean(1, keepdim=True).\
                        mul(-1).expand_as(param)
                param.add_(negMean)

    def clampConvParams(self): #entender depois mas basicamente limita os  pesos para 1 e -1
        for param in self.target_modules:
            with torch.no_grad():
                param.copy_(param.clamp(-1.0, 1.0))

    def save_params(self): #copia os pesos em target_module para saved_params
        for index, target in enumerate(self.target_modules):
            with torch.no_grad():
                self.saved_params[index].copy_(target) #copia inplace target_module para saved_params

    def binarizeConvParams(self): # calcula o alpha 
        with torch.no_grad():
            for param in self.target_modules:
                saidas = param[0].nelement() #num de saídas
                dim = param.size() #dimensões
                # print(f"saídas:{saidas}")
                # print(f"dimensão:{dim}")
                if len(dim) == 4: #conlucional = [saída, entrada, altura_kernel, largura_kernel]
                    alpha = param.abs().mean(dim=(1,2,3), keepdim=True).expand(dim) #soma dos valores absolutos
                elif len(dim) == 2: #linear = [entrada, saída]
                    alpha = param.abs().mean(dim=1, keepdim=True).expand(dim)
                # param = param.sign().mul(alpha.expand(dim)) não é inplace, vai so mudar a copia
                param.copy_(param.sign().mul(alpha)) #type: ignore

    def restore(self):
        for index in range(self.num_of_params):
            with torch.no_grad():
                self.target_modules[index].copy_(self.saved_params[index]) #copia de saved_params para target_modules

    def updateBinaryGradWeight(self):
        with torch.no_grad():
            for param in self.target_modules:
                saidas = param[0].nelement() #num de saídas
                dim = param.size() #dimensões
                # print(f"saídas:{saidas}")
                # print(f"dimensão:{dim}")
                if len(dim) == 4:
                    alpha = param.abs()\
                            .mean(dim=(1,2,3), keepdim=True)\
                            .expand(dim).clone()
                elif len(dim) == 2:
                    alpha = param.abs().mean(1, keepdim=True).expand(dim).clone()

                alpha[param.lt(-1.0)] = 0 #type: ignore
                alpha[param.gt(1.0)] = 0 #type: ignore

                alpha.mul_(param.grad) #type: ignore #alpha * gradiente dos pesos
                
                alpha_add = param.grad.div(saidas)

                param.grad = alpha.add(alpha_add) #type: ignore



def plot_batch(dataloader, classes):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    title=' '.join(classes[label] for label in labels)
    npimg = torchvision.utils.make_grid(images).numpy()
    plt.figure(figsize=(10,4))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

