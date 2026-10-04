from perceptron import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
from typing import cast
from sklearn.utils import Bunch

iris = cast(Bunch, load_iris())
X = iris.data[:, (0, 1)]
y = (iris.target == 0).astype(np.int_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

perceptron = Perceptron(0.001, 100)

perceptron.fit(X_train, y_train)

pred = perceptron.predict(X_test)

accuracy_score(pred, y_test)

report = classification_report(pred, y_test, digits=2)
print(report)
