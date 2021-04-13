import os
import glob

class ConstTree(object):
    def __init__(self):
        self.left = None
        self.right = None

    def size(self):
        self.size = 1
        if self.left is not None:
            self.size += self.left.size()
        if self.right is not None:
            self.size += self.right.size()
        return self.size

    def set_spans(self):
        if self.word is not None:
            self.span = self.word
            return self.span

        self.span = self.left.set_spans()
        if self.right is not None:
            self.span += ' ' + self.right.set_spans()
        return self.span

    # 为树中的每个节点找到对应label
    def get_labels(self, spans, labels, dictionary):
        if self.span in dictionary:
            spans[self.idx] = self.span
            labels[self.idx] = dictionary[self.span]
        if self.left is not None:
            self.left.get_labels(spans, labels, dictionary)
        if self.right is not None:
            self.right.get_labels(spans, labels, dictionary)

# 处理分割好的句子
def load_sentences(dirpath):
    sents = []
    with open(os.path.join(dirpath, 'SOStr.txt')) as sentsfile:
        for line in sentsfile:
            sent = ' '.join(line.split('|'))
            sents.append(sent.strip())
    return sents

# 处理train/dev/test分割
def load_splits(dirpath):
    splits = []
    with open(os.path.join(dirpath, 'datasetSplit.txt')) as splitfile:
        #读掉第一行title
        splitfile.readline()
        for line in splitfile:
            idx, split = line.split(',')
            splits.append(int(split))
    return splits


def load_parents(dirpath):
    parents = []
    with open(os.path.join(dirpath, 'STree.txt')) as parentsfile:
        for line in parentsfile:
            p = ' '.join(line.split('|'))
            parents.append(p.strip())
    return parents

# 处理每个短语的打分
# 处理字典
def load_dictionary(dirpath):
    labels = []
    with open(os.path.join(dirpath, 'sentiment_labels.txt')) as labelsfile:
        labelsfile.readline()
        for line in labelsfile:
            idx, rating = line.split('|')
            idx = int(idx)
            rating = float(rating)
            if rating <= 0.2:
                label = -2
            elif rating <= 0.4:
                label = -1
            elif rating > 0.8:
                label = +2
            elif rating > 0.6:
                label = +1
            else:
                label = 0
            labels.append(label)
    
    # key: str; value: label
    d = {}
    with open(os.path.join(dirpath, 'dictionary.txt')) as dictionary:
        for line in dictionary:
            s, idx = line.split('|')
            d[s] = labels[int(idx)]
    return d

def build_vocab(filepaths, dst_path, lowercase = True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    
    with open(dst_path, 'w') as f:
        for word in sorted(vocab):
            f.write(word + '\n')

# 将sentence, 以及对应的STree分到train/test/dev中
def split(sst_dir, train_dir, dev_dir, test_dir):
    sents = load_sentences(sst_dir)
    splits = load_splits(sst_dir)
    parents = load_parents(sst_dir)

    with open(os.path.join(train_dir, 'sents.txt'), 'w') as train, \
         open(os.path.join(dev_dir, 'sents.txt'), 'w') as dev, \
         open(os.path.join(test_dir, 'sents.txt'), 'w') as test, \
         open(os.path.join(train_dir, 'parents.txt'), 'w') as trainparents, \
         open(os.path.join(dev_dir, 'parents.txt'), 'w') as devparents, \
         open(os.path.join(test_dir, 'parents.txt'), 'w') as testparents:

        for sent, split, p in zip(sents, splits, parents):
            if split == 1:
                train.write(sent)
                train.write('\n')
                trainparents.write(p)
                trainparents.write('\n')
            elif split == 2:
                test.write(sent)
                test.write('\n')
                testparents.write(p)
                testparents.write('\n')
            else:
                dev.write(sent)
                dev.write('\n')
                devparents.write(p)
                devparents.write('\n')


def load_constituency_tree(parents, words):
    trees = []
    root = None
    size = len(parents)
    for i in range(size):
        trees.append(None)
    for i in range(size):
        if not trees[i]:
            idx = i
            prev = None
            prev_idx = None
            word = words[i]
            while True:
                tree = ConstTree()
                parent = parents[idx] - 1
                tree.word, tree.parent, tree.idx = word, parent, idx
                word = None
                if prev is not None:
                    if tree.left is None:
                        tree.left = prev
                    else:
                        tree.right = prev
                trees[idx] = tree
                if parent >= 0 and trees[parent] is not None:
                    if trees[parent].left is None:
                        trees[parent].left = tree
                    else:
                        trees[parent].right = tree
                    break
                elif parent == -1:
                    root = tree
                    break
                else:
                    prev = tree
                    prev_idx = idx
                    idx = parent
    return root


def load_trees(dirpath):
    const_trees, toks = [], []
    with open(os.path.join(dirpath, 'parents.txt')) as parentsfile, \
         open(os.path.join(dirpath, 'sents.txt')) as toksfile:
        parents = []
        for line in parentsfile:
            parents.append(list(map(int, line.strip().split())))
        print("parents" + str(len(parents)))
        for line in toksfile:
            toks.append(line.strip().split())
        print("toks" + str(len(toks)))
        for i in range(len(toks)):
            const_trees.append(load_constituency_tree(parents[i], toks[i]))
    return const_trees, toks

def get_labels(tree, dictionary):
    size = tree.size()
    spans, labels = [], []
    for i in range(size):
        labels.append(None)
        spans.append(None)
    tree.get_labels(spans, labels, dictionary)
    return spans, labels

def write_labels(dirpath, dictionary):
    print('Writing labels for tree in ' + dirpath)
    with open(os.path.join(dirpath, 'labels.txt'), 'w') as labels:
        const_trees, toks = load_trees(dirpath)

        # write span labels
        for i in range(len(const_trees)):
            const_trees[i].set_spans()
            # const tree labels
            s, l = [], []
            for j in range(const_trees[i].size()):
                s.append(None)
                l.append(None)
            const_trees[i].get_labels(s, l, dictionary)
            labels.write(' '.join(map(str, l)) + '\n')

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing Stanford Sentiment Treebank...')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    lib_dir = os.path.join(base_dir, 'lib')
    sst_dir = os.path.join(data_dir, 'sst')
    train_dir = os.path.join(sst_dir, 'train')
    dev_dir = os.path.join(sst_dir, 'dev')
    test_dir = os.path.join(sst_dir, 'test')
    make_dirs([train_dir, dev_dir, test_dir])

    split(sst_dir, train_dir, dev_dir, test_dir)
    sent_paths = glob.glob(os.path.join(sst_dir, '*/sents.txt'))

    # build vocabulary
    build_vocab(sent_paths, os.path.join(sst_dir, 'vocab.txt'))
    build_vocab(sent_paths, os.path.join(sst_dir, 'vocab-cased.txt'), lowercase=False)

    # write sentiment labels for nodes in trees
    dictionary = load_dictionary(sst_dir)
    write_labels(train_dir, dictionary)
    write_labels(dev_dir, dictionary)
    write_labels(test_dir, dictionary)
