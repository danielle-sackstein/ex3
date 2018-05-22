import numpy as np
from matplotlib.pyplot import *
from sklearn.linear_model import LogisticRegression


def read_all_data(file_name):
    with open(file_name) as file_:
        lines = file_.readlines()

        data = []
        labels = []

        for line in lines:
            values = line.split()
            data.append(np.array(values[0:-1]))
            labels.append(values[-1])

    return to_ndarray(data), to_ndarray(labels)

def to_ndarray(list_):
    return np.array(list_).astype(np.float64)

def split_data_randomly_indexes(total_size, test_size):
    # create the train_indexes has having all the indexes
    # and the test_indexes as being empty.
    # We will move test_size indexes from train_indexes to test_indexes

    train_indexes = [i for i in range(total_size)]
    test_indexes = []

    for i in range(test_size):
        # we need to remove one of the indexes in train_indexes.
        # but note that train_indexes may contain gaps because we have removed some of its original values.
        # So for instance, if train_indexes now contains [0, 55, 2345, 20000]
        # and we need to remove one more we select a position between 0 and 3 - let's say 1
        # and then we remove 55 which is in place 1 (and move it to test_indexes

        select_position_of_index_to_move = np.random.randint(0, len(train_indexes))
        index_to_move = train_indexes[select_position_of_index_to_move]
        train_indexes.remove(index_to_move)
        test_indexes.append(index_to_move)

    return train_indexes, test_indexes


def split_data_randomly(all_data, all_labels, test_size):
    train_indexes, test_indexes = split_data_randomly_indexes(all_data.shape[0], test_size)
    return split_data(all_data, all_labels, test_indexes, train_indexes)


def split_data(all_data, all_labels, test_indexes, train_indexes):
    train_data = [all_data[i] for i in train_indexes]
    train_labels = [all_labels[i] for i in train_indexes]
    test_data = [all_data[i] for i in test_indexes]
    test_labels = [all_labels[i] for i in test_indexes]

    return to_ndarray(train_data), to_ndarray(train_labels), to_ndarray(test_data), to_ndarray(test_labels)


def FindPositivesForProb(sorted_probs, min_prob):
    total_count = len(sorted_probs)
    for i in range(total_count):
        if sorted_probs[i] > min_prob:
            return total_count - i
    return 0


def get_data(test_size):
    all_data, all_labels = read_all_data('spam.data')
    return split_data_randomly(all_data, all_labels, test_size)


if __name__ == "__main__":

    test_size = 1000
    iteration_count = 10

    train_data, train_labels, test_data, test_labels = get_data(test_size)

    NP = len([label for label in test_labels if label == 1])
    NN = test_size - NP

    average_roc_x = [0] * NP
    average_roc_y = [0] * NP

    for it in range(iteration_count):

        lgr = LogisticRegression()
        lgr.fit(train_data, train_labels)

        probs = lgr.predict_proba(test_data)[:,1]
        indexes = np.argsort(probs)

        sorted_probs = [probs[i] for i in indexes]
        sorted_positive_probs = [probs[i] for i in indexes if test_labels[i] == 1]

        for i in range(NP): # the number of TP required
            min_prob = sorted_positive_probs[-i-1]
            Ni = FindPositivesForProb (sorted_probs, min_prob)
            TPR = i / float(NP)
            FPR = (Ni - i) / float(NN)

            average_roc_x[i] += FPR
            average_roc_y[i] += TPR

    average_roc_x = [v/float(iteration_count) for v in average_roc_x]
    average_roc_x.insert(0, 0)
    average_roc_x.append(1)

    average_roc_y = [v/float(iteration_count) for v in average_roc_y]
    average_roc_y.insert(0, 0)
    average_roc_y.append(1)

    plot(average_roc_x, average_roc_y, linestyle='-', label='ROC')

    ylabel('TPR')
    xlabel('FPR')

    title('ROC')
    savefig('roc.png')
    clf()
