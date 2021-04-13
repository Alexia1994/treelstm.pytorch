# -*- coding: utf-8 -*-
from __future__ import print_function
from hyperparams import Hyperparams as hp
import numpy as np
import codecs
import os
#import regex
from collections import Counter
import pickle
import itertools
import json
# 拿到了所有的单词及其计数
def make_vocab(train_path, test_path, vocab):
    '''Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed/fname`
    '''  
    f_train = json.load(open(train_path, 'r'))
    f_test = json.load(open(test_path, 'r'))

    total_text = []
    for train_line in f_train:
        if "text" not in train_line:
            continue
        text = []
        equation = []
        text = train_line["text"].strip().split()
        text = [word.lower() for word in text]
        equation = train_line["equations"].replace("\r\n", "").strip().split()
        equation = [word.lower() for word in equation]
        total_text.extend(text)
        total_text.extend(equation)


    for test_line in f_test:

        if "text" not in test_line:
            continue
        text = []
        equation = []
        text = test_line["text"].strip().split()
        text = [word.lower() for word in text]
        equation = train_line["equations"].replace("\r\n", "").strip().split()
        equation = [word.lower() for word in equation]
        total_text.extend(text)
        total_text.extend(equation)
    print(len(total_text))
    word2cnt = Counter(total_text) # 单词计数
  
    if not os.path.exists('data'):
        os.mkdir('data')

    with codecs.open('data/{}'.format(vocab), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>","pseudo_root"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            #fout.write(u"{}\t{}\n".format(word, cnt))
            fout.write("{}\t{}\n".format(word, cnt))
        fout.close()
if __name__ == '__main__':
    make_vocab(hp.train, hp.test, "all_vocab.tsv")
    print("Done")