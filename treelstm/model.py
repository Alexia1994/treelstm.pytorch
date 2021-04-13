import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Constants
from . import utils

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        # dim = 0 means sum all rows up
        # 1 * mem_dim 
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        # 1 * 3*mem_dim + 1 * 3*mem_dim
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        # repeat代表：行上复制len(child_h)次，列上复制1次
        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        # 递归调用
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            # 新建输入
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            # lambda在定义时被直接调用
            # map()方法, 第一个参数是一个方法的引用 然后是可以有多个可迭代对象, 将后面的可迭代对象按序拆包
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        # treee.state其实是根节点的c, h
        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = torch.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out), dim=1)
        return out


# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)

    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        lstate, lhidden = self.childsumtreelstm(ltree, linputs)
        rstate, rhidden = self.childsumtreelstm(rtree, rinputs)
        output = self.similarity(lstate, rstate)
        return output


################################################################
# module for n_arytreelstm
class N_aryTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, criterion, device, n):
        super(N_aryTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.n = n
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)

        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        
        self.output_module = None
        self.criterion = criterion
        self.device = device
    
    def set_output_module(self, output_module):
        self.output_module = output_module

    # def node_forward(self, inputs, child_c, child_h):

    #     child_h_linear_sum = torch.sum(self.iouh(child_h), dim=0, keepdim=True)
        
    #     iou = self.ioux(inputs) + child_h_linear_sum
    #     i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
    #     i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        
    #     # a little bit ugly
    #     # N*mem_dim -> 1*mem_dim
    #     f = []
    #     for idx in range(self.n):
    #         f.append(torch.sigmoid(
    #             torch.sum(self.fh(child_h), dim=0, keepdim=True) +
    #             self.fx(inputs))
    #         )
    #     f = torch.cat(f, dim=0)
    #     fc = torch.mul(f, child_c)

    #     c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
    #     h = torch.mul(o, torch.tanh(c))
    #     return c, h


    def node_forward(self, child_c, child_h):
        child_h_linear_sum = torch.sum(self.iouh(child_h), dim=0, keepdim=True)
        
        iou = child_h_linear_sum
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        
        # a little bit ugly
        # N*mem_dim -> 1*mem_dim
        f = []
        for idx in range(self.n):
            f.append(torch.sigmoid(torch.sum(self.fh(child_h), dim=0, keepdim=True)))
        
        f = torch.cat(f, dim=0)
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def leaf_node_forward(self, inputs):
        c = self.fx(inputs)
        o = torch.sigmoid(self.fx(inputs))
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        # need fix: 或许需要转移到device上
        loss = torch.zeros(1)
        for idx in range(tree.num_children):
            _, child_loss = self.forward(tree.children[idx], inputs)
            loss += child_loss
        
        if tree.num_children == 0:
            tree.state = self.leaf_node_forward(inputs[tree.idx-1].unsqueeze(0))
        else:
            # cat(dim=0) 是上下合并(下一个拼接到上一个的下边)
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            # maybe need fix
            tree.state = self.node_forward(child_c, child_h)
        
        # 计算输出
        if self.output_module != None:
            output = self.output_module.forward(tree.state[1])
            tree.output = output
            if tree.gold_label != None:
                target = utils.map_label_to_target_sentiment(tree.gold_label)
                target = target.to(self.device)
                loss = loss + self.criterion(output, target)
        return tree.state, loss

        
# 应该返回一个装有每个节点label的list
class Sentiment(nn.Module):
    def __init__(self, mem_dim, num_classes):
        super(Sentiment, self).__init__()
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.l1 = nn.Linear(self.mem_dim, self.num_classes)

    def forward(self, h):
        out = F.log_softmax(self.l1(h), dim=1)
        return out

# putting the whole model together
class SentimentTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, num_classes, freeze, criterion, device, dropout, n):
        super(SentimentTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)
        if freeze:
            self.emb.weight.requires_grad = False
        self.n_arytreelstm = N_aryTreeLSTM(in_dim, mem_dim, criterion, device, n)
        self.sentiment = Sentiment(mem_dim, num_classes)
        self.n_arytreelstm.set_output_module(self.sentiment)

    def forward(self, tree, inputs):
        embedded = self.dropout(self.emb(inputs))
        tree.state, loss = self.n_arytreelstm(tree, embedded)
        out = tree.output
        return out, loss
