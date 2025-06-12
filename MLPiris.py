import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as prep

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(data.describe())

output = data['Species']
labelEncoder = prep.LabelEncoder()

output = labelEncoder.fit_transform(output).astype(np.float64)
input = data.drop(columns=['Species'])

inputTrain, inputTest, outputTrain, outputTest = train_test_split(input, output, random_state=1)

neuralNetwork = MLPClassifier(
                                verbose = True,
                                random_state = 1,
                                learning_rate_init = 0.001,
                                max_iter = 1000
                             )

neuralNetwork.fit(inputTrain, outputTrain)
print("Modelo preparado para o teste!")

outputPred = neuralNetwork.predict(inputTest)

score = accuracy_score(outputTest, outputPred)
print(f"Acurácia: {score*100:.2f}%")