import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

# np.random.seed(999)
# random.seed(999)

curr_id_col = 0
label_col = 1
next_id_col = 2
data_cols = range(6, 134)
image_hight = 16
image_width = 8
min_score = -9999999

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
    accum_preds = []
    test_accuracy = 0

    # get rel data
    x_test = test.iloc[:, data_cols].values
    y_test = test.iloc[:, label_col].values
    next_id_test = test.iloc[:, next_id_col].values
    for x, y, next_id in zip(x_test, y_test, next_id_test):

        # predict
        y_hat = np.argmax(np.dot(w, x))
        if y_hat == char_to_idx[y]:
            test_accuracy += 1

        accum_preds.append(idx_to_char[y_hat])

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

            # update params
            w = w + phi(x, char_to_idx[y]) - phi(x, max_idx)

            # log accuracy
            if max_idx == char_to_idx[y]:
                train_accuracy += 1

        print("Multi-Class Structured Perceptron train accuracy: " + str(train_accuracy / len(x_train)))
        train_accuracy = 0

    # test
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

        if char_to_idx[y] == max_idx:
            test_accuracy += 1

        accum_preds.append(idx_to_char[max_idx])

    accuracy = test_accuracy/len(test.index)
    print("Multi-Class Structured Perceptron test accuracy: " + str(accuracy))
    path = "./output/multiclass_structured_perceptron/multiclass_structured_perceptron_" + str(round(accuracy, 4)) + ".csv"
    write_predictions(path, accum_preds)


# from dataframe to list of lists where each value is a tuple
def get_seq_as_list(data):

    sequnces = []
    x = data.iloc[:, data_cols].values
    y = data.iloc[:, label_col].values
    next_id_char = data.iloc[:, next_id_col].values

    seq = []
    for ocr, label, next_id in zip(x, y, next_id_char):
        # create tuple of x and y values and append to list
        seq.append((ocr, label))

        # on the last char append the sequence to the sequences list
        if next_id == -1:
            sequnces.append(seq)
            seq = []

    return sequnces


def viterbi(seq, w, num_eng_chars, char_to_idx, is_train=True):

    def phi(x, y, prev_char_idx):
        vec = np.zeros(num_eng_chars * num_params_per_class + num_eng_chars*num_eng_chars)
        x_start_idx = y*num_params_per_class  # start place for coping vector
        x_end_idx = x_start_idx + num_params_per_class  # end place for coping vector
        vec[x_start_idx:x_end_idx] = x
        bigram_idx = num_eng_chars * num_params_per_class + y * num_eng_chars + prev_char_idx  # bigram indicator placing
        vec[bigram_idx] = 1
        return vec

    num_params_per_class = image_hight * image_width

    # initialization of data structures
    seq_len = len(seq)
    score_matrix = np.zeros((seq_len, num_eng_chars), dtype=int)  # save the score of the max path until each cell
    index_matrix = np.zeros((seq_len, num_eng_chars), dtype=int)  # save the best previous char
    y_hat = np.zeros(seq_len, dtype=int)  # the index of the max path
    dollar_idx = char_to_idx["$"]  # the index of the special token

    # find most probable sequence
    # for the first row
    x = seq[0][0]
    for curr_char in range(num_eng_chars):
        score_matrix[0, curr_char] = np.dot(w, np.transpose(phi(x, curr_char, dollar_idx)))  # per each possible char get the score
        index_matrix[0, curr_char] = dollar_idx  # the previous char is always the $ sign

    # recursion step
    for row_idx in range(1, seq_len):
        x = seq[row_idx][0]
        for curr_char in range(num_eng_chars):
            max_char_idx = -1
            max_score = -1
            for prev_char in range(num_eng_chars):
                s = np.dot(w, np.transpose(phi(x, curr_char, prev_char))) + score_matrix[row_idx-1, prev_char]  # per each possible char get the score
                if s > max_score or prev_char == 0:
                    max_score = s
                    max_char_idx = prev_char

            score_matrix[row_idx, curr_char] = max_score  # save max score
            index_matrix[row_idx, curr_char] = max_char_idx  # save the prev char that generated that max score

    # backtrack
    best_final_char_idx = np.argmax(score_matrix[seq_len-1, :])
    y_hat[seq_len-1] = best_final_char_idx
    for char_in_word_idx in range(seq_len-2, -1, -1):
        y_hat[char_in_word_idx] = index_matrix[char_in_word_idx+1, y_hat[char_in_word_idx+1]]

    # update params in case of training
    if is_train:
        # get labels
        labels = [char_to_idx["$"]] + [char_to_idx[y] for x, y in seq]
        idx = 0
        for curr_x, curr_y in seq:
            prev_pred_char_idx = y_hat[idx-1] if idx > 0 else dollar_idx
            w += phi(curr_x, char_to_idx[curr_y], labels[idx]) - phi(curr_x, y_hat[idx], prev_pred_char_idx)
            idx += 1

    return y_hat, w


def plot_w(path, w):

    w = w.reshape((27, 27))[:-1, :-1]  # reshape w to matrix and remove the last row and the last column corresponding to the $ sign
    w = np.transpose(w)  # transpose so rows will be prev char and columns current char
    ticks = list("abcdefghijklmnopqrstuvwxyz")
    sns.heatmap(w, cmap='Greys', cbar=False, xticklabels=ticks, yticklabels=ticks)
    plt.yticks(rotation=360)
    plt.savefig(path)
    plt.close()


# multi-class structured perceptron
def multiclass_structured_perceptron_bigram(train, test, epochs, char_to_idx, idx_to_char):

    print("\n")

    num_params_per_class = image_hight * image_width
    num_eng_chars = len(char_to_idx)

    # init weights: one weight vector per character and a 26*27 vector for bigram indicator
    w = np.zeros(num_eng_chars * num_params_per_class + num_eng_chars*num_eng_chars)
    start_time = time.time()

    # get the train set as a list of lists. each internal list is a sequence
    train_set = get_seq_as_list(train)

    # train
    for epoch in range(epochs):

        # in each epoch reshuffle the set - the internal order of sequences remain but the the order of sequences is reordered
        random.shuffle(train_set)
        train_accuracy = 0
        for idx, seq in enumerate(train_set):

            # get labels
            labels = [char_to_idx[y] for x, y in seq]
            Y = np.asarray(labels)

            # get the most likely sequence
            y_hat, w = viterbi(seq, w.copy(), num_eng_chars, char_to_idx, True)

            # calc accuracy
            train_accuracy += (Y == y_hat).sum()

        elapsed_time = time.time() - start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(str(time_str) + "\tMulti-Class Structured Perceptron Bigram train accuracy: " + str(train_accuracy / len(train.index)))

    # test
    accum_preds = []
    test_accuracy = 0

    # get the train set as a list of lists. each internal list is a sequence
    test_set = get_seq_as_list(test)

    for seq in test_set:
        # get labels
        labels = [char_to_idx[y] for x, y in seq]
        Y = np.asarray(labels)

        # get the most likely sequence
        y_hat, _ = viterbi(seq, w, num_eng_chars, char_to_idx, False)

        # calc accuracy
        test_accuracy += (Y == y_hat).sum()

        # save predictions
        seq_as_list = y_hat.tolist()
        for char_idx in seq_as_list:
            accum_preds.append(idx_to_char[char_idx])

    accuracy = test_accuracy / len(test.index)

    elapsed_time = time.time() - start_time
    time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    print(str(time_str) + "\tMulti-Class Structured Perceptron Bigram test accuracy: " + str(accuracy))
    path = "./output/multiclass_structured_perceptron_bigram/multiclass_structured_perceptron_bigram_" + str(round(accuracy, 4)) + ".csv"
    write_predictions(path, accum_preds)
    save_w_path = "./output/multiclass_structured_perceptron_bigram/multiclass_structured_perceptron_bigram_" + str(round(accuracy, 4)) + ".png"
    plot_w(save_w_path, w[num_eng_chars * num_params_per_class:].copy())


# read data files
def run(train_path, test_path, epochs):

    heder_names = ["id", "letter", "next_id", "word_id", "position", "fold"] + ["x_" + str(i) for i in range(image_hight*image_width)]
    train = pd.read_csv(train_path, sep="\t", header=None, names=heder_names, index_col=False)
    test = pd.read_csv(test_path, sep="\t", header=None, names=heder_names, index_col=False)
    chars = list("abcdefghijklmnopqrstuvwxyz")
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    #  multiclass_perceptron(train, test, epochs, char_to_idx, idx_to_char)
    #  multiclass_structured_perceptron(train, test, epochs, char_to_idx, idx_to_char)

    char_to_idx["$"] = 26
    idx_to_char[26] = "$"
    multiclass_structured_perceptron_bigram(train, test, epochs, char_to_idx, idx_to_char)


if __name__ == '__main__':

    train_path = "/home/idan/Desktop/studies/Advanced_Techniques_in_Machine_Learning/ex2/data/letters.train.data"
    test_path = "/home/idan/Desktop/studies/Advanced_Techniques_in_Machine_Learning/ex2/data/letters.test.data"
    epochs = 5
    check_accuracy = True

    # create directory for writing results
    path = "./output/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "multiclass_perceptron/", exist_ok=True)
    os.makedirs(path + "multiclass_structured_perceptron/", exist_ok=True)
    os.makedirs(path + "multiclass_structured_perceptron_bigram/", exist_ok=True)

    #for i in range(20):
    #    run(train_path, test_path, epochs)


    if check_accuracy:

        x1 = pd.read_csv("/home/idan/Desktop/studies/Advanced_Techniques_in_Machine_Learning/ex2/output/test_labels.csv", header=None, index_col=False, names=["x1"])
        x2 = pd.read_csv("/home/idan/Desktop/studies/Advanced_Techniques_in_Machine_Learning/ex2/output/multiclass_structured_perceptron_bigram/multiclass_structured_perceptron_bigram_0.7810.csv", header=None, index_col=False, names=["x2"])

        x1ToList = x1['x1'].tolist()
        x2ToList = x2['x2'].tolist()

        check = 0
        for c1, c2 in zip(x1ToList, x2ToList):
            if c1 == c2:
                check += 1

        print(check)