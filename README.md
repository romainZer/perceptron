# Perceptron implementation

From-stratch implementation of the perceptron, the simplest neural network, in Python.
[Linked article](https://www.quarkml.com/2022/12/perceptron-algorithm-understanding-and-implementation-python)

## Installation

Clone the repository. Then,

```bash
cd perceptron

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The script generates a dataset, trains the perceptron and prints the resulting accuracy.

For example :

```plaintext
              precision    recall  f1-score   support

         0.0       0.93      1.00      0.97        43
         1.0       1.00      0.91      0.95        32

    accuracy                           0.96        75
   macro avg       0.97      0.95      0.96        75
weighted avg       0.96      0.96      0.96        75
```
