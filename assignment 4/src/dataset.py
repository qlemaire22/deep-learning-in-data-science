from constants import *
import numpy as np

def read_data():
    book_data = ''
    with open(DATA_FILENAME, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        # book_data += line.replace('\n', '').replace('\t', '')
        book_data += line


    book_char = []

    for i in range(len(book_data)):
        if not(book_data[i] in book_char):
            book_char.append(book_data[i])

    print(len(book_char))

    return book_data, book_char, len(book_char)

def char_to_ind(char, book_char):
    alphabet_size = len(book_char)
    ind = np.zeros((alphabet_size, 1), dtype=int)
    ind[book_char.index(char)] = 1
    return ind.T

def ind_to_char(ind, book_char):
    return book_char[np.argmax(ind)]
