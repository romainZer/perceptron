import numpy as np


class Perceptron:
    def __init__(self, learning_rate, epochs) -> None:
        self.weights = np.zeros(0)
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, z) -> np.ndarray:
        """
        Fonction d'activation seuil (heaviside)
        """
        return np.heaviside(z, 0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fonction d'entraînement du perceptron
        """
        n_features = X.shape[1]

        self.weights = np.zeros((n_features))
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                # somme pondérée des entrées et ajout du biais
                z = np.dot(X, self.weights) + self.bias
                y_pred = self.activation(z)

                self.weights = self.weights = (
                    self.weights + self.learning_rate * (y[i] - y_pred[i]) * X[i]
                )
                self.bias = self.bias + self.learning_rate * (y[i] - y_pred[i])

        return self.weights, self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
