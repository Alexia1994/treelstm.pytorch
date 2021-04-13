from copy import deepcopy

import torch


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        return torch.mean(torch.mul(x, y))

    def mse(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        return torch.mean((x - y) ** 2)
<<<<<<< HEAD
    
    def sentiment_accuracy_score(self, predictions, labels, fine_gained = True):
        correct = (predictions==labels).sum()
        total = labels.size(0)
        acc = float(correct)/total
        return acc
=======
>>>>>>> 228a314add09fc7f39ea752aa7b1fcf756cfe277
