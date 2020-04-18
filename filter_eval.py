import numpy as np


def filter_punct(str1):
    punct = [" ,", " .", " ?", " !", " :", " \"", " \'", " /"]
    for p in punct:
        f = ""
        for s in str1.split(p):
            f = f + s
        str1 = f
    return str1


def formalize(str1):
    punct = [",", ".", "?", "!", ":", "\"", "\'", "/"]
    # add space in front of each punctuation
    for p in punct:
        f = ""
        for s in str1.split(p):
            f = f + s + " " + p
        str1 = f[:-1]
    # filter the double space in the sentence
    f = ""
    for s in str1.split(" "):
        if s != "":
            f = f + s + " "
    str1 = f[:-1]
    # change all the characters to the lowercase
    str1 = str1.lower()
    return str1


def edit_distance(sentence1, sentence2):
    matrix = [[i + j for j in range(len(sentence2) + 1)] for i in range(len(sentence1) + 1)]

    for i in range(1, len(sentence1) + 1):
        for j in range(1, len(sentence2) + 1):
            if sentence1[i - 1] == sentence2[j - 1]:
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    
    return matrix[len(sentence1)][len(sentence2)]


def ldp(sentence1, sentence2):
    return np.exp(- abs(len(sentence1) - len(sentence2)) / len(sentence1))
