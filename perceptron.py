import numpy as np
from matplotlib.pyplot import plot, legend, ylabel, xlabel, title, savefig, clf
from sklearn.svm import SVC


class Perceptron:
    max_iterations = 10000

    def __init__(self):
        self._w = []

    def fit(self, X, y):

        d = X.shape[1]
        self._w = [0] * d

        for i in range(self.max_iterations):
            if self.all_labels_correct(X, y):
                break

    def all_labels_correct(self, _X, _y):
        _m = _X.shape[0]
        for i in range(_m):
            if _y[i] * np.inner(self._w, _X[i]) <= 0:
                self._w += _y[i] * _X[i]
                return False
        return True

    def predict(self, x):
        product = np.inner(self._w, x)
        return 1 if np.any(product > 0) else -1

    def score(self, data, labels):
        sample_count = data.shape[0]
        correct_label_count = 0
        for i in range(sample_count):
            predicted_label = self.predict(data[i])
            if predicted_label == labels[i]:
                correct_label_count += 1
        return correct_label_count / sample_count


def classify(w, sample):
    inner_product = np.inner(w, sample)
    return 1 if np.any(inner_product > 0) else -1


def get_labeled_data(sample_count, distribution):
    while True:
        data, labels, sum_labels = distribution(sample_count)

        all_equal = (sum_labels == sample_count) or (sum_labels == -sample_count)
        if not all_equal:
            break
    return data, np.array(labels)


dist_mean = [0] * 2
dist_cov = np.identity(2)


def D1(sample_count):
    data = np.random.multivariate_normal(dist_mean, dist_cov, sample_count)
    labels = []
    w = (0.3, -0.5)
    sum_labels = 0
    for i in range(sample_count):
        label = classify(w, data[i])
        labels.append(label)
        sum_labels += label
    return data, labels, sum_labels


rectangles = {
    -1: [[-3, 1], [1, 3]],
    1: [[-1, 3], [-3, -1]]
}


def D2(sample_count):
    data = []
    labels = []
    sum_labels = 0
    for i in range(sample_count):
        choose_rect = 2 * np.random.randint(0, 2) - 1
        rect = rectangles[choose_rect]
        x_range = rect[0]
        y_range = rect[1]

        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])

        label = choose_rect

        sum_labels += label

        data.append(np.array([x, y]))
        labels.append(label)

    return np.array(data), np.array(labels), sum_labels


def get_scores(train_data, train_labels, test_data, test_labels):
    pcn = Perceptron()
    pcn.fit(train_data, train_labels)
    pcn_score = pcn.score(test_data, test_labels)

    svm = SVC(C=1e10, kernel='linear')
    svm.fit(train_data, train_labels)
    svm_score = svm.score(test_data, test_labels)

    return pcn_score, svm_score


def get_mean_scores(train_count, test_count, iteration_count, distribution):
    pcn_scores = []
    svm_scores = []

    for iteration in range(iteration_count):
        train_data, train_labels = get_labeled_data(train_count, distribution)
        test_data, test_labels = get_labeled_data(test_count, distribution)

        pcn_score, svm_score = get_scores(
            train_data, train_labels,
            test_data, test_labels)

        pcn_scores.append(pcn_score)
        svm_scores.append(svm_score)

    return np.average(pcn_scores), np.average(svm_scores)


def calculate_and_plot_accuracies(k, M, distribution, distribution_name):
    accuracies_pcn, accuracies_svm = calculate_accuracies(M, k, distribution)
    plot_accuracies(M, accuracies_pcn, accuracies_svm, distribution_name)


def calculate_accuracies(M, k, distribution):
    accuracies_pcn = []
    accuracies_svm = []

    for m in M:
        pcn_accuracy, svm_accuracy = get_mean_scores(
            m, k, 500, distribution)

        accuracies_pcn.append(pcn_accuracy)
        accuracies_svm.append(svm_accuracy)

    return accuracies_pcn, accuracies_svm


def plot_accuracies(M, accuracies_pcn, accuracies_svm, distribution_name):
    plot_pcn, = plot(M, accuracies_pcn, linestyle='-', label='perceptron')
    plot_svm, = plot(M, accuracies_svm, linestyle='--', label='SVM')

    legend(handles=[plot_pcn, plot_svm])

    ylabel('Accuracy')
    xlabel('Number of samples')

    title('Accuracy vs m for distribution {}'.format(distribution_name))
    savefig('AccuracyVsM.{}.png'.format(distribution_name))
    clf()


if __name__ == "__main__":
    k = 10000
    M = [5, 10, 15, 25, 70]

    calculate_and_plot_accuracies(k, M, D1, "D1");
    calculate_and_plot_accuracies(k, M, D2, "D2");
