import numpy as np
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

    return np.array(data), np.array(labels)

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

    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)


if __name__ == "__main__":
    all_data, all_labels = read_all_data('spam.data')

    test_size = 1000

    train_data, train_labels, test_data, test_labels = split_data_randomly(all_data, all_labels, test_size)

    lgr = LogisticRegression()
    lgr.fit(train_data, train_labels)

    prob_pos = lgr.predict_proba(test_data)


    pass