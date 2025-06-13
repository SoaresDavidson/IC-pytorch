# Binarização da LeNet-5 com PyTorch

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow.svg)

Este repositório contém o código-fonte e os resultados do projeto de Iniciação Científica focado na implementação e análise de uma versão binarizada da rede neural LeNet-5, utilizando o framework PyTorch e o dataset MNIST.

## 📖 Sumário

* [Visão Geral do Projeto](#-visão-geral-do-projeto)
* [Conceitos Fundamentais](#-conceitos-fundamentais)
* [Estrutura do Repositório](#-estrutura-do-repositório)
* [Como Executar](#-como-executar)
* [Trabalhos Futuros](#-trabalhos-futuros)

---

## 🔭 Visão Geral do Projeto

O objetivo deste trabalho é explorar as Redes Neurais Binarizadas (BNNs) como uma técnica de compressão e otimização de modelos. Implementamos uma versão binarizada da LeNet-5, uma arquitetura clássica para reconhecimento de dígitos, e a treinamos no dataset MNIST. A análise foca na comparação entre o modelo binarizado e sua contraparte de precisão total em termos de acurácia, tamanho do modelo e velocidade de inferência (simulada).

---

## 🧠 Conceitos Fundamentais

#### LeNet-5
Uma das primeiras Redes Neurais Convolucionais (CNNs) de sucesso, proposta por Yann LeCun. Sua arquitetura é um marco no campo da Visão Computacional, sendo altamente eficaz para tarefas como o reconhecimento de dígitos manuscritos.

#### Redes Neurais Binarizadas (BNNs)
BNNs são redes neurais onde os pesos e/ou as ativações são restringidos a apenas dois valores: **+1** ou **-1**. Essa restrição drástica oferece duas vantagens principais:
* **Redução de Memória:** Cada peso pode ser armazenado com apenas 1 bit, resultando em uma compressão de até 32x em comparação com modelos de 32 bits.
* **Eficiência Computacional:** Multiplicações de ponto flutuante, que são caras, podem ser substituídas por operações de bit `XNOR`, que são extremamente rápidas em hardware especializado.

#### Straight-Through Estimator (STE)
A função de binarização (`sign(x)`) possui gradiente zero em quase todos os pontos, o que impede o treinamento através de backpropagation. O STE é uma técnica que contorna esse problema:
* **Na passagem `forward`:** Usa-se a função `sign` normalmente.
* **Na passagem `backward`:** O gradiente é "enganado" ou aproximado, geralmente usando a derivada de uma função proxy como a identidade (`f(x) = x`) ou uma função de clipping, permitindo que o erro flua através da rede e os pesos sejam atualizados.

---

## 📁 Estrutura do Repositório

```
.
├── data/                  # Diretório para os datasets
├── main.py                # Script principal para treino e avaliação do modelo
├── requirements.txt       # Lista de dependências do projeto
└── README.md              # Este arquivo
```

---

## 📝 Trabalhos Futuros

- [x] Implementar a binarização das camadas convolucionais.
- [x] Implementar a binarização das camadas totalmente conectadas.
- [x] Binarizar as ativações, além dos pesos.
- [ ] Experimentar diferentes funções de gradiente para o STE.
- [ ] Analisar o impacto da ordem das camadas (ex: `BN -> Ativação` vs. `Ativação -> BN`).
- [ ] Exportar o modelo para o formato ONNX para implantação em hardware de ponta.
