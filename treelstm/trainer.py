from tqdm import tqdm
<<<<<<< HEAD
import torch
from . import utils
"""
For Sentiment Module
"""

class SentimentTrainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(SentimentTrainer, self).__init__()
=======

import torch

from . import utils


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
>>>>>>> 228a314add09fc7f39ea752aa7b1fcf756cfe277
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
<<<<<<< HEAD
        # 导数清零
        self.optimizer.zero_grad()
        total_loss = 0.0
        # randperm 随机打乱序列
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        # 每一个例子
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, input, label = dataset[indices[idx]]
            # 一颗有n个叶节点的树最后有 2n+1 个节点，因此应有 2n+1 个target
            # utils.print_tree(tree, 0)
            input = input.to(self.device)
            out, loss = self.model(tree, input)
            # 因为返回的是Variable
            total_loss += loss.data[0]
            loss.backward()

=======
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            ltree, linput, rtree, rinput, label = dataset[indices[idx]]
            target = utils.map_label_to_target(label, dataset.num_classes)
            linput, rinput = linput.to(self.device), rinput.to(self.device)
            target = target.to(self.device)
            output = self.model(ltree, linput, rtree, rinput)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            loss.backward()
>>>>>>> 228a314add09fc7f39ea752aa7b1fcf756cfe277
            if idx % self.args.batchsize == 0 and idx > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
<<<<<<< HEAD

            for idx in tqdm(range(len(dataset)),desc='Testing epoch  ' + str(self.epoch) + ''):
                tree, input, label = dataset[idx]
                target = utils.map_label_to_target_sentiment(label, dataset.num_classes, fine_grain=self.args.fine_grain)
                input = input.to(self.device)
                target = target.to(self.device)
                output, _ = self.model(tree, input)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                # python 切片表达式包含两个“:”，用于分隔三个参数(start_index: end_index: step)
                output[:,1] = -9999 # no need middle (neutral) value
                # 返回每一行中最大值的那个元素，且返回其索引
                val, pred = torch.max(output, 1)
                predictions[idx] = pred.data.cpu()[0]

        return loss/len(dataset), predictions
=======
            indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                ltree, linput, rtree, rinput, label = dataset[idx]
                target = utils.map_label_to_target(label, dataset.num_classes)
                linput, rinput = linput.to(self.device), rinput.to(self.device)
                target = target.to(self.device)
                output = self.model(ltree, linput, rtree, rinput)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                output = output.squeeze().to('cpu')
                predictions[idx] = torch.dot(indices, torch.exp(output))
        return total_loss / len(dataset), predictions
>>>>>>> 228a314add09fc7f39ea752aa7b1fcf756cfe277
