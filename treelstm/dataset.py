import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

from . import Constants
from .tree import Tree


# Dataset class for SICK dataset
class SICKDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(SICKDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.lsentences = self.read_sentences(os.path.join(path, 'a.toks'))
        self.rsentences = self.read_sentences(os.path.join(path, 'b.toks'))

        self.ltrees = self.read_trees(os.path.join(path, 'a.parents'))
        self.rtrees = self.read_trees(os.path.join(path, 'b.parents'))

        self.labels = self.read_labels(os.path.join(path, 'sim.txt'))

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return (ltree, lsent, rtree, rsent, label)

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.tensor(indices, dtype=torch.long, device='cpu')

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.tensor(labels, dtype=torch.float, device='cpu')
        return labels
<<<<<<< HEAD


# Dataset class for Stanford Sentiment Treebank Dataset
class SSTDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes, fine_grain):
        super(SSTDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes
        self.fine_grain = fine_grain

        temp_sentences = self.read_sentences(os.path.join(path, 'sents.txt'))
        temp_trees = self.read_trees(os.path.join(path, 'parents.txt'), os.path.join(path, 'labels.txt'))

        # self.labels = self.read_labels(os.path.join(path,'dlabels.txt'))
        self.labels = []

        if not self.fine_grain:
            # only get pos or neg
            new_trees = []
            new_sentences = []
            for i in range(len(temp_trees)):
                if temp_trees[i].gold_label != 1: # 0 neg, 1 neutral, 2 pos
                    new_trees.append(temp_trees[i])
                    new_sentences.append(temp_sentences[i])
            self.trees = new_trees
            self.sentences = new_sentences
        else:
            self.trees = temp_trees
            self.sentences = temp_sentences

        for i in range(0, len(self.trees)):
            self.labels.append(self.trees[i].gold_label)
        self.labels = torch.Tensor(self.labels) # let labels be tensor
        self.size = len(self.trees)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # ltree = deepcopy(self.ltrees[index])
        # rtree = deepcopy(self.rtrees[index])
        # lsent = deepcopy(self.lsentences[index])
        # rsent = deepcopy(self.rsentences[index])
        # label = deepcopy(self.labels[index])
        tree = deepcopy(self.trees[index])
        sent = deepcopy(self.sentences[index])
        label = deepcopy(self.labels[index])
        return (tree, sent, label)

    def read_sentences(self, filename):
        with open(filename,'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, filename_parents, filename_labels):
        pfile = open(filename_parents, 'r') # parent node
        lfile = open(filename_labels, 'r') # label node
        p = pfile.readlines()
        l = lfile.readlines()
        pl = zip(p, l) # (parent, label) tuple
        trees = [self.read_tree(p_line, l_line) for p_line, l_line in tqdm(pl)]

        return trees

    def parse_label_token(self, x):
        if x == '#':
            return None
        else:
            if self.fine_grain: # -2 -1 0 1 2 => 0 1 2 3 4
                return int(x)+2
            else: # # -2 -1 0 1 2 => 0 1 2
                tmp = int(x)
                if tmp < 0:
                    return 0
                elif tmp == 0:
                    return 1
                elif tmp >0 :
                    return 2

    def read_tree(self, parent_line, label_line):
        # FIXED: tree.idx, also tree dict() use base 1 as it was in dataset
        # parents is list base 0, keep idx-1
        # labels is list base 0, keep idx-1
        parents = list(map(int,parent_line.split()))
        trees = dict()
        root = None
        labels = list(map(self.parse_label_token, label_line.split()))
        for i in range(1, len(parents)+1):
            if i not in trees.keys() and parents[i-1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx] = tree
                    tree.idx = idx # -1 remove -1 here to prevent embs[tree.idx -1] = -1 while tree.idx = 0
                    tree.gold_label = labels[idx-1] # add node label
                    #if trees[parent-1] is not None:
                    if parent in trees.keys():
                        trees[parent].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        # Not in used
        with open(filename,'r') as f:
            labels = map(lambda x: float(x), f.readlines())
            labels = torch.Tensor(labels)
        return labels
=======
>>>>>>> 228a314add09fc7f39ea752aa7b1fcf756cfe277
