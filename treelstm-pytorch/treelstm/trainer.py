from tqdm import tqdm
import torch
from . import utils
"""
For Sentiment Module
"""

class SentimentTrainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        # 导数清零
        self.optimizer.zero_grad()
        total_loss = 0.0
        # randperm 随机打乱序列
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        # 每一个例子
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, sent, label = dataset[indices[idx]]
            input = Var(sent)
            target = Var(map_label_to_target_sentiment(label, dataset.num_classes, fine_grain = self.args.fine_grain))

            # emb = F.torch.unsqueeze(self.embedding_model(input), 1)
            output = self.model.forward(tree, input)
            # need fix
            loss = self.criterion(output, target)
            total_loss += loss.data[0]
            loss.backward()

            if idx % self.args.batchsize == 0 and idx > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        #predictions = predictions
        indices = torch.range(1, dataset.num_classes)
        for idx in tqdm(range(len(dataset)),desc='Testing epoch  ' + str(self.epoch) + ''):
            tree, sent, label = dataset[idx]
            input = Var(sent, volatile=True)
            target = Var(map_label_to_target_sentiment(label,dataset.num_classes, fine_grain=self.args.fine_grain), volatile=True)

            output, _ = self.model(tree, emb) # size(1,5)
            err = self.criterion(output, target)
            loss += err.data[0]
            output[:,1] = -9999 # no need middle (neutral) value
            val, pred = torch.max(output, 1)
            #predictions[idx] = pred.data.cpu()[0][0]
            predictions[idx] = pred.data.cpu()[0]
            # predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/len(dataset), predictions