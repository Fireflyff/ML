import numpy as np
from sklearn import metrics


class yy_LR:
    def __init__(self):
        self.weight = None

    def _sigmoid(self, x):
        return 1./(1+np.exp(-x))

    def _cost(self, y_hat, y_true):
        return -np.sum(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)) / len(y_hat)

    def dx(self, weight, X, y):

        return X.T.dot(self._sigmoid(X.dot(weight)) - y) / len(y)

    def _gradient_descent(self, X, y, lr, n_iter, epsilon):
        """
        :param X: data
        :param y: label
        :param lr: learning rate
        :param n_iter:
        :param epsilon: early stop
        :return:
        """
        # init weight
        weight = np.zeros(X.shape[1])
        for i in range(n_iter):
            d = self.dx(weight, X, y)
            old_weight = weight
            weight = weight - lr * d

            if abs(self._cost(self._sigmoid(X.dot(old_weight)), y) - self._cost(self._sigmoid(X.dot(weight)), y)) < epsilon:
                break
        return weight

    def fit(self, data, label, lr=1e-2, n_iter=100, epsilon=1e-3):
        X = np.hstack([np.ones((len(data), 1)), data])
        self.weight = self._gradient_descent(X, label, lr, n_iter, epsilon)

        return self

    def predict(self, data):
        X = np.hstack([np.ones((len(data), 1)), data])
        return self._sigmoid(X.dot(self.weight))

    def score(self, data, y):
        y_proba = self.predict(data)
        auc = metrics.roc_auc_score(y, y_proba)
        acc = metrics.accuracy_score(y, np.array(y_proba >= 0.5, dtype=int))
        return auc, acc


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y < 2, :]
    y = y[y < 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    my_lr = yy_LR()
    my_lr.fit(X_train, y_train)

    print(my_lr.score(X_test, y_test))



