import numpy as np
from sklearn.naive_bayes import GaussianNB
import string


class AtbashCipherEncoder:
    def __init__(self):
        self.classifier = GaussianNB()

    @staticmethod
    def letters_array_to_numbers_array(letters_array):
        return list(map((lambda x: ord(x)), letters_array))

    def predict_result(self, features, labels, text_to_encode):
        self.classifier.fit(features, labels)

        array = [ord(char) for char in text_to_encode.lower()]

        prediction = self.classifier.predict(np.array(array).reshape(-1, 1))

        result_list = prediction.tolist()

        for index, char in enumerate(text_to_encode):
            if char.isupper():
                result_list[index] = result_list[index].upper()

        return "".join(result_list)

    def encode(self, text_to_encode):
        alphabet_letters = list(string.ascii_lowercase)

        features = np.array(AtbashCipherEncoder.letters_array_to_numbers_array(alphabet_letters)).reshape(-1, 1)
        labels = np.array(alphabet_letters[::-1])

        return self.predict_result(features, labels, text_to_encode)


encoder = AtbashCipherEncoder()
print(encoder.encode("ALESSANDRO"))
