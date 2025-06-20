import torch.nn as nn
import torch
import numpy

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
            negMean = param.data.mean(1, keepdim=True).\
                    mul(-1).expand_as(param.data)
            param.data = param.data.add(negMean)

    def clampConvParams(self): #entender depois mas basicamente limita os  pesos para 1 e -1
        for param in self.target_modules:
            param.data = \
                    param.data.clamp(-1.0, 1.0)

    def save_params(self): #copia os pesos em target_module para saved_params
        for index, target in enumerate(self.target_modules):
            self.saved_params[index].copy_(target.data) #copia inplace target_module para saved_params

    def binarizeConvParams(self): # calcula o alpha 
        for param in self.target_modules:
            n = param.data[0].nelement() #num de saídas
            s = param.data.size() #dimensões
            # print(f"n:{n}")
            # print(f"s:{s}")
            if len(s) == 4: #conlucional = [saída, entrada, altura_kernel, largura_kernel]
                m = param.data.abs()\
                        .sum(3, keepdim=True)\
                        .sum(2, keepdim=True)\
                        .sum(1, keepdim=True)\
                        .div(n) #soma dos valores absolutos
            elif len(s) == 2: #linear = [entrada, saída]
                m = param.data.abs().sum(1, keepdim=True).div(n)

            param.data = param.data.sign().mul(m.expand(s)) #type: ignore

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index]) #copia de saved_params para target_modules

    def updateBinaryGradWeight(self):
        for param in self.target_modules:
            weight = param.data
            n = weight[0].nelement() # num de saídas 
            s = weight.size() #dimensões
            if len(s) == 4:
                m = weight.abs()\
                        .sum(3, keepdim=True)\
                        .sum(2, keepdim=True)\
                        .sum(1, keepdim=True)\
                        .div(n).expand(s).clone() #norm deprecated
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s).clone()

            m[weight.lt(-1.0)] = 0 #type: ignore
            m[weight.gt(1.0)] = 0 #type: ignore

            m = m.mul(param.grad.data) #type: ignore
            m_add = weight.sign().mul(param.grad.data)

            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True)\
                        .sum(1, keepdim=True)\
                        .div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
                
            m_add = m_add.mul(weight.sign())
            param.grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
