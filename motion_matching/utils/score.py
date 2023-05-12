import math
import numpy as np


class Score(object):
    # To be replaced with a better scoring algorithm, if found in the future
    def percentage_score(self, score):
        percentage = 100 - (score * 100)
        return int(percentage)

    def cosine_distance(self, A, B):
        A = A.reshape(2,)
        B = B.reshape(2,)
        cosine = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        return self.percentage_score(math.sqrt(2 * (1 - cosine)))

    def normalize(self, input_test):
        for k in range(0, 17):
            input_test[:, k] = input_test[:, k] / \
                np.linalg.norm(input_test[:, k])
        return input_test

    def compare(self, ip, model):
        # ip = self.normalize(ip)

        # print(ip.shape)
        # print(model.shape)
        if ip.shape == model.shape:
            scores = []
            for k in range(0, 6):
                scores.append(self.cosine_distance(ip[:, k], model[:, k]))
            return np.mean(scores), scores
        else:
            if not ip.shape:
                print('skip because human pose')
            else:
                print('skip because robot pose')
