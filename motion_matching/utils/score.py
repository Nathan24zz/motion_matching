import math
import numpy as np


class Score(object):
    # To be replaced with a better scoring
    # algorithm, if found in the future
    def percentage_score(self, score):
        percentage = 100 - (score * 100)
        return int(percentage)

    def cosine_distance(self, A, B):
        # reduce dimension ([[x, y]] -> [x, y])
        A = A.reshape(2,)
        B = B.reshape(2,)

        nominator = np.dot(A, B)
        denominator = np.linalg.norm(A) * np.linalg.norm(B)

        if denominator == 0:
            cosine = 0
        else:
            cosine = nominator / denominator
        return self.percentage_score(math.sqrt(2 * (1 - cosine)))

    def normalize(self, input_test):
        for k in range(0, 6):
            nominator = input_test[:, k]
            denominator = np.linalg.norm(input_test[:, k])

            if denominator == 0:
                input_test[:, k] = [[0, 0]]
            else:
                input_test[:, k] = nominator / denominator
        return input_test

    def compare(self, ip, model):
        ip = ip.astype('float32')
        model = model.astype('float32')
        ip = self.normalize(ip)
        model = self.normalize(model)

        # print(ip.shape)
        # print(model.shape)
        if ip.shape == model.shape:
            scores = []
            for k in range(0, 6):
                scores.append(abs(self.cosine_distance(
                    ip[:, k], model[:, k])))
            # ignore head and trunk
            return np.mean(scores[2:]), scores[2:]
        else:
            if not ip.shape:
                print('skip because human pose')
            else:
                print('skip because robot pose')
