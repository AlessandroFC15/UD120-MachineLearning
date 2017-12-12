import numpy as np
from sklearn.naive_bayes import GaussianNB
import string


def letters_array_to_numbers_array(lettersArray):
    return list(map((lambda x: ord(x)), lettersArray))


def predict_result(classifier, string):
    array = [ord(char) for char in string.lower()]

    prediction = classifier.predict(np.array(array).reshape(-1, 1))

    result_list = prediction.tolist()

    for index, char in enumerate(string):
        if char.isupper():
            result_list[index] = result_list[index].upper()

    return "".join(result_list)


alphabet_letters = list(string.ascii_lowercase)

X = np.array(letters_array_to_numbers_array(alphabet_letters)).reshape(-1, 1)
Y = np.array(alphabet_letters[::-1])

clf = GaussianNB()
clf.fit(X, Y)

string_to_test = "ALESSANDRO"

result = predict_result(clf, string_to_test)

print(result)
