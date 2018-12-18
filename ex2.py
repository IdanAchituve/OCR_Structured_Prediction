import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os

np.random.seed(999)

curr_id_col = 0
label_col = 1
next_id_col = 2
data_cols = range(6, 134)
image_hight = 16
image_width = 8


def print_image(x):

    img_vec = np.reshape(x, (16, 8)).astype(float)
    plt.imshow(img_vec, cmap='gray')
    plt.show()


def get_x_y_shuffled(data):

    x = data.iloc[:, data_cols].values
    y = data.iloc[:, label_col].values
    next_id_char = data.iloc[:, next_id_col].values
    x_rand, y_rand, next_id_char_rand = shuffle(x, y, next_id_char)

    return x_rand, y_rand, next_id_char_rand


def write_predictions(path, preds):
    with open(path, 'w') as f:
        for idx, item in enumerate(preds):
            if idx != len(preds) - 1:
                f.write("%s\n" % item)
            else:
                f.write("%s" % item)


# multi-class perceptron
def multiclass_perceptron(train, test, epochs, char_to_idx, idx_to_char):


    # init weights
    w = np.zeros((len(char_to_idx), image_hight*image_width))

    # train
    train_accuracy = 0  # check accuracy at the char level
    for epoch in range(epochs):
        # in each epoch reshuffle the set
        x_train, y_train, _ = get_x_y_shuffled(train)
        for x, y in zip(x_train, y_train):
            y_hat = np.argmax(np.dot(w, x))
            # update if prediction differ from label
            if y_hat != char_to_idx[y]:
                w[char_to_idx[y], :] = w[char_to_idx[y], :] + x
                w[y_hat, :] = w[y_hat, :] - x
            else:
                train_accuracy += 1

        print("Multi-Class Perceptron train accuracy: " + str(train_accuracy / len(x_train)))
        train_accuracy = 0

    # test
    labels = preds = np.array([])
    accum_preds = []
    test_accuracy = 0

    # get rel data
    x_test = test.iloc[:, data_cols].values
    y_test = test.iloc[:, label_col].values
    next_id_test = test.iloc[:, next_id_col].values
    for x, y, next_id in zip(x_test, y_test, next_id_test):

        # predict
        y_hat = np.argmax(np.dot(w, x))

        labels = np.append(labels, char_to_idx[y])
        preds = np.append(preds, y_hat)
        accum_preds.append(idx_to_char[y_hat])

        # at the end of each sequence check accuracy
        if next_id == -1:
            test_accuracy += np.array_equal(labels, preds)
            labels = preds = np.array([])

    accuracy = test_accuracy/len(test.index)
    print("Multi-Class Perceptron test accuracy: " + str(accuracy))
    path = "./output/multiclass_perceptron/multiclass_perceptron_" + str(round(accuracy, 4)) + ".csv"
    write_predictions(path, accum_preds)


# multi-class structured perceptron
def multiclass_structured_perceptron(train, test, epochs, char_to_idx, idx_to_char):

    def phi(x, y):
        vec = np.zeros((len(char_to_idx), num_params_per_class))
        vec[y] = x
        return vec.flatten()

    print("\n")
    num_params_per_class = image_hight*image_width
    # init weights as 1 flat vector
    w = np.zeros((len(char_to_idx), num_params_per_class)).flatten()

    # train
    train_accuracy = 0  # check accuracy at the char level
    for epoch in range(epochs):
        # in each epoch reshuffle the set
        x_train, y_train, _ = get_x_y_shuffled(train)
        for x, y in zip(x_train, y_train):
            max_y_hat = 0.0
            max_idx = 0
            for char, idx in char_to_idx.items():
                y_hat = np.dot(w, np.transpose(phi(x, idx)))
                if y_hat > max_y_hat:
                    max_y_hat = y_hat
                    max_idx = idx

            w = w + phi(x, char_to_idx[y]) - phi(x, max_idx)
            if max_idx == char_to_idx[y]:
                train_accuracy += 1

        print("Multi-Class Structured Perceptron train accuracy: " + str(train_accuracy / len(x_train)))
        train_accuracy = 0

    # test
    labels = preds = np.array([])
    accum_preds = []
    test_accuracy = 0

    # get rel data
    x_test = test.iloc[:, data_cols].values
    y_test = test.iloc[:, label_col].values
    next_id_test = test.iloc[:, next_id_col].values
    for x, y, next_id in zip(x_test, y_test, next_id_test):

        # predict
        max_y_hat = 0.0
        max_idx = 0
        for char, idx in char_to_idx.items():
            y_hat = np.dot(w, np.transpose(phi(x, idx)))
            if y_hat > max_y_hat:
                max_y_hat = y_hat
                max_idx = idx

        labels = np.append(labels, char_to_idx[y])
        preds = np.append(preds, max_idx)
        accum_preds.append(idx_to_char[max_idx])

        # at the end of each sequence check accuracy
        if next_id == -1:
            test_accuracy += np.array_equal(labels, preds)
            labels = preds = np.array([])

    accuracy = test_accuracy/len(test.index)
    print("Multi-Class Structured Perceptron test accuracy: " + str(accuracy))
    path = "./output/multiclass_structured_perceptron/multiclass_structured_perceptron_" + str(round(accuracy, 4)) + ".csv"
    write_predictions(path, accum_preds)


# multi-class structured perceptron
def multiclass_structured_perceptron_bigram(train, test, epochs, char_to_idx, idx_to_char):

    def phi(x, y, prev_char_idx):
        vec = np.zeros(num_chars * num_params_per_class + num_chars*len(char_to_idx))
        place_x_start = y*num_params_per_class  # start place for coping vector
        place_x_end = place_x_start + num_params_per_class  # end place for coping vector
        place_bigram_ind = place_x_end + y*len(char_to_idx) + prev_char_idx  # bigram indicator placing
        vec[place_x_start:place_x_end] = x
        vec[place_bigram_ind] = 1
        return vec

    print("\n")
    num_params_per_class = image_hight*image_width
    num_chars = len(char_to_idx) - 1
    # init weights: one weight vector per character and a 26*27 vector for bigram indicator
    w = np.zeros(num_chars * num_params_per_class + num_chars*len(char_to_idx))

    # train
    train_accuracy = 0  # check accuracy at the char level
    for epoch in range(epochs):
        # in each epoch reshuffle the set
        x_train, y_train, next_id_train = get_x_y_shuffled(train)
        prev_char = "$"
        for x, y, next_id in zip(x_train, y_train, next_id_train):
            max_y_hat = 0.0
            max_idx = 0
            for char, idx in char_to_idx.items():
                y_hat = np.dot(w, np.transpose(phi(x, idx)))
                if y_hat > max_y_hat:
                    max_y_hat = y_hat
                    max_idx = idx

            w = w + phi(x, char_to_idx[y]) - phi(x, max_idx)
            if max_idx == char_to_idx[y]:
                train_accuracy += 1

        print("Multi-Class Structured Perceptron train accuracy: " + str(train_accuracy / len(x_train)))
        train_accuracy = 0

    # test
    labels = preds = np.array([])
    accum_preds = []
    test_accuracy = 0

    # get rel data
    x_test = test.iloc[:, data_cols].values
    y_test = test.iloc[:, label_col].values
    next_id_test = test.iloc[:, next_id_col].values
    for x, y, next_id in zip(x_test, y_test, next_id_test):

        # predict
        max_y_hat = 0.0
        max_idx = 0
        for char, idx in char_to_idx.items():
            y_hat = np.dot(w, np.transpose(phi(x, idx)))
            if y_hat > max_y_hat:
                max_y_hat = y_hat
                max_idx = idx

        labels = np.append(labels, char_to_idx[y])
        preds = np.append(preds, max_idx)
        accum_preds.append(idx_to_char[max_idx])

        # at the end of each sequence check accuracy
        if next_id == -1:
            test_accuracy += np.array_equal(labels, preds)
            labels = preds = np.array([])

    accuracy = test_accuracy/len(test.index)
    print("Multi-Class Structured Perceptron test accuracy: " + str(accuracy))
    path = "./output/multiclass_structured_perceptron/multiclass_structured_perceptron_" + str(round(accuracy, 4)) + ".csv"
    write_predictions(path, accum_preds)


# read data files
def run(train_path, test_path, epochs):

    train = pd.read_csv(train_path, sep="\t", header=None)
    test = pd.read_csv(test_path, sep="\t", header=None)
    chars = list("abcdefghijklmnopqrstuvwxyz")
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    multiclass_perceptron(train, test, epochs, char_to_idx, idx_to_char)
    multiclass_structured_perceptron(train, test, epochs, char_to_idx, idx_to_char)

    char_to_idx["$"] = 26
    idx_to_char[26] = "$"


if __name__ == '__main__':

    train_path = "/home/idan/Desktop/studies/Advanced_Techniques_in_Machine_Learning/ex2/data/letters.train.data"
    test_path = "/home/idan/Desktop/studies/Advanced_Techniques_in_Machine_Learning/ex2/data/letters.test.data"
    epochs = 6

    # create directory for writing results
    path = "./output/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "multiclass_perceptron/", exist_ok=True)
    os.makedirs(path + "multiclass_structured_perceptron/", exist_ok=True)
    os.makedirs(path + "multiclass_structured_perceptron_bigram/", exist_ok=True)

    run(train_path, test_path, epochs)