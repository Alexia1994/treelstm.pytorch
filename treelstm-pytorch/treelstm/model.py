import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Constants

################################################################
# module for n_arytreelstm
class N_aryTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, n):
        super(N_aryTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.n = n
        self.h_state = []
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)

        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):

        child_h_linear_sum = torch.sum(self.iouh(child_h), dim=0, keepdim=True)
        
        iou = self.ioux(inputs) + child_h_linear_sum
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)
        
        # a little bit ugly
        # N*mem_dim -> 1*mem_dim
        for idx in range(self.n):
            f[idx] = F.sigmoid(
                torch.sum(self.fh(child_h), dim=0, keepdim=True) +
                self.fx(inputs)
            )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)
        
        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        _, h = tree.state
        self.h_state.append(h)
        return tree.state
    
    def cal_loss(self, tree, target):
        
# 应该返回一个装有每个节点label的list
class Sentiment(nn.Module):
    def __init__(self, mem_dim, num_classes):
        super(SentimentOutput, self).__init__()
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.l1 = nn.Linear(self.mem_dim, self.num_classes)

    def forward(self, h_state):
        out = []
        for h in h_state:
            out.append(F.log_softmax(self.l1(h)))
        return out

# putting the whole model together
class SentimentTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, num_classes, sparsity, freeze, n):
        super(SentimentTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.n_arytreelstm = N_aryTreeLSTM(in_dim, mem_dim, n)
        self.sentiment = Sentiment(mem_dim, num_classes)

    def forward(self, tree, inputs):
        # 返回最后的root的state
        state = self.n_arytreelstm(tree, inputs)
        # 拿到一个本树所有节点的输出的list
        output = self.sentiment(self.n_arytreelstm.h_state)
        return output
