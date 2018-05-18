import numpy as np
from matplotlib.pyplot import plot, legend, ylabel, xlabel, title, savefig
from sklearn.svm import SVC

k = 10000
M = {5, 10, 15, 25, 70}


class Perceptron:

    def __init__(self):
        self._w = []

    def fit(self, X, y):
        d = X.shape[1]
        m = X.shape[0]

        self._w = [0] * d
        i_exists = True

        while i_exists:

            # check if there exists such i
            i_exists = False
            for i in range(m):
                if y[i] * np.inner(self._w, X[i]) <= 0:
                    self._w += + y[i] * X[i]
                    i_exists = True
                    break

    def predict(self, x):
        product = np.inner(self._w, x)
        if np.any(product > 0):
            return 1
        return -1


if __name__ == "__main__":

    # D1 - train (x1,..xm)
    accuracies_percpetron = [0] * 500
    accuracies_SVM = [0] * 500

    for p, m in enumerate(range(len(M))):
        for iteration in range(500):

            # get X
            mean = [0] * 2
            cov = np.identity(2)
            X = np.random.multivariate_normal(mean, cov, (m, 2))

            # classify
            y = [0] * m
            w = (0.3, -0.5)

            # get y
            for j in range(m):
                product = np.inner(w, X[j])
                if np.any(product > 0):
                    y[j] = 1
                else:
                    y[j] = -1

            # D1 - test(z1,..zk)
            mean = [0] * 2
            cov = np.identity(2)
            Z = np.random.multivariate_normal(mean, cov, (k, 2))

            # classify - perceptron
            perceptron = Perceptron()
            perceptron.fit(X, y)

            for j in range(k):
                predict = perceptron.predict(Z[j])
                if predict == y[j]:
                    accuracies_percpetron[iteration] += 1

            # classify - SVM + get accuracy
            clf = SVC(C=1e10, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
            clf.fit(X, y)
            accuracies_SVM[iteration] = int(clf.score(X, y))

        # get accuracy - perceptron
        for iteration in range(500):
            accuracies_percpetron[iteration] /= 500

        accuracies_percpetron, = plot(m, accuracies_percpetron, linestyle='-', label='percpetron')
        accuracies_SVM, = plot(m, accuracies_SVM, linestyle='--', label='SVM')

        legend(handles=[accuracies_percpetron, accuracies_SVM])

        ylabel('Accuracy')
        xlabel('Number of samples')
        title('Accuracy vs m = {}'.format(m))
        savefig('AcurracyVsM.{}.png'.format(m))
        clf()
