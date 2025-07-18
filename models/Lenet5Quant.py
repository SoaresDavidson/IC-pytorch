# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# import brevitas.nn as qnn
# from brevitas.quant import Int32Bias


# # --- Model ---
# class Lenet5Quant(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.quant_inp = qnn.QuantIdentity(bit_width=4, return_quant_tensor=True)
#         self.conv1 = qnn.QuantConv2d(1, 6, 5, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
#         self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
#         self.conv2 = qnn.QuantConv2d(6, 16, 5, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
#         self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
#         self.fc1 = qnn.QuantLinear(16 * 5 * 5, 120, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
#         self.relu3 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
#         self.fc2 = qnn.QuantLinear(120, 84, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
#         self.relu4 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
#         self.fc3 = qnn.QuantLinear(84, 10, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

#     def forward(self, x):
#         out = self.quant_inp(x)
#         out = self.relu1(self.conv1(out))
#         out = F.max_pool2d(out, 2)
#         out = self.relu2(self.conv2(out))
#         out = F.max_pool2d(out, 2)
#         out = out.reshape(out.shape[0], -1)
#         out = self.relu3(self.fc1(out))
#         out = self.relu4(self.fc2(out))
#         out = self.fc3(out)
#         return out
