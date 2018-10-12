import constants
import numpy as np
import tqdm


def read_file_name(filename, validation_filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    with open(validation_filename, 'r') as f:
        lines_ind = f.readlines()

    validation_indexes = lines_ind[0][:-1].split(' ')
    validation_indexes = list(map(int, validation_indexes))

    dataset = {}
    names = []
    labels = []
    dataset["names_train"] = []
    dataset["labels_train"] = []
    dataset["names_validation"] = []
    dataset["labels_validation"] = []

    all_names = ""

    index = 0

    for line in lines:
        temp = line.replace(',', '').lower().split(' ')
        name = ""
        for i in range(len(temp) - 1):
            if i != 0:
                name += ' '
            name += temp[i]
            all_names += temp[i]
        temp = temp[-1].replace('\n', '')

        names.append(name)
        labels.append(int(temp))

        if (index + 1) in validation_indexes:
            dataset["names_validation"].append(name)
            dataset["labels_validation"].append(int(temp)-1)
        else:
            dataset["names_train"].append(name)
            dataset["labels_train"].append(int(temp)-1)

        index += 1

    dataset["alphabet"] = ''.join(set(all_names)) + ' '
    dataset["d"] = len(dataset["alphabet"])
    dataset["K"] = len(list(set(labels)))
    dataset["n_len"] = len(max(names, key=len))

    dataset["labels_validation"] = np.array(dataset["labels_validation"])
    dataset["labels_train"] = np.array(dataset["labels_train"])

    return dataset


def create_one_hot(dataset):
    one_hot_array = np.zeros(
        (dataset["d"] * dataset["n_len"], len(dataset["names_train"])))

    char_to_int = dict((c, i) for i, c in enumerate(dataset["alphabet"]))
    index = 0
    for name in dataset["names_train"]:
        one_hot = np.zeros((dataset["d"], dataset["n_len"]))
        integer_encoded = [char_to_int[char] for char in name]
        i = 0
        for value in integer_encoded:
            letter = np.zeros((dataset["d"]))
            letter[value] = 1
            one_hot[:, i] = letter
            i += 1
        one_hot_array[:dataset["d"]*dataset["n_len"], index] = one_hot.flatten('F')
        index += 1

    dataset["one_hot_train"] = one_hot_array

    one_hot_array = np.zeros(
        (dataset["d"] * dataset["n_len"], len(dataset["names_validation"])))

    char_to_int = dict((c, i) for i, c in enumerate(dataset["alphabet"]))
    index = 0

    for name in dataset["names_validation"]:
        one_hot = np.zeros((dataset["d"], dataset["n_len"]))
        integer_encoded = [char_to_int[char] for char in name]
        i = 0
        for value in integer_encoded:
            letter = np.zeros((dataset["d"]))
            letter[value] = 1
            one_hot[:, i] = letter
            i += 1
        one_hot_array[:dataset["d"]*dataset["n_len"], index] = one_hot.flatten('F')
        index += 1

    dataset["one_hot_validation"] = one_hot_array

    dataset["names_friends"] = ["lemaire", "yeramian", "vecchio", "del castillo", "larsson", "blain", "vinesse", "thiron", "blanchard", "gogoulou", "sebahi", "do"]
    one_hot_array = np.zeros(
        (dataset["d"] * dataset["n_len"], len(dataset["names_friends"])))
    char_to_int = dict((c, i) for i, c in enumerate(dataset["alphabet"]))
    index = 0

    for name in dataset["names_friends"]:
        one_hot = np.zeros((dataset["d"], dataset["n_len"]))
        integer_encoded = [char_to_int[char] for char in name]
        i = 0
        for value in integer_encoded:
            letter = np.zeros((dataset["d"]))
            letter[value] = 1
            one_hot[:, i] = letter
            i += 1
        one_hot_array[:dataset["d"]*dataset["n_len"], index] = one_hot.flatten('F')
        index += 1

    dataset["one_hot_friends"] = one_hot_array

    dataset["one_hot_label_train"], dataset["balance_train"] = createOneHotLabels(dataset["labels_train"], len(dataset["labels_train"]), dataset["K"])
    dataset["one_hot_label_validation"], dataset["balance_validation"] = createOneHotLabels(dataset["labels_validation"], len(dataset["labels_validation"]), dataset["K"])
    dataset["one_hot_label_train"] = dataset["one_hot_label_train"].T
    dataset["one_hot_label_validation"] = dataset["one_hot_label_validation"].T
    return dataset

def createOneHotLabels(labels, N, K):
    balance = np.zeros(K, dtype=int)

    one_hot_labels = np.zeros((N, K))
    for i in range(len(labels)):
        balance[labels[i]] += 1
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels, balance

def getDataset():
    dataset = read_file_name(constants.name_file, constants.validation_file)
    dataset = create_one_hot(dataset)
    return dataset
