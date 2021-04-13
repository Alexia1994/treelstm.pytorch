# -*- coding: utf-8 -*-
from hyperparams import Hyperparams as hp
import numpy as np
import codecs
import pickle
import json
import random
from tree import Tree
from cal import middle_to_after, preorder, inorder
ops_rule = {
    '=': 0,
    '+': 1,
    '-': 1,
    '*': 2,
    '/': 2,
    '^': 3
}

def build_tree(expression):
    expression = middle_to_after(expression)
    stack  = []
    for item in expression:
        if item[0] in ops_rule:
            right = stack.pop()
            left = stack.pop()
            node = Tree(item[0], item[1])
            node.left = left
            node.right = right
            stack.append(node)
        else:
            node = Tree(item[0], item[1])
            stack.append(node)

    if len(stack) != 1:
        #print(expression)
        return "wrong_tree"
    root = stack[0]
    return root

def load_vocab():
    
    vocab = [line.split()[0] for line in codecs.open('data/all_vocab.tsv', 'r', 'utf-8').readlines()if int(line.split()[1])>=hp.min_cnt]
    
    ######### vocab max word num
    vocab = vocab[:hp.max_word]
    #tag_to_index = {"number" : 0, "unknown_number" : 1, "symbol" : 2}
    #########
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    return word2idx, idx2word

def get_word_vector(path, embedding_size):
    cnt = 0
    word2idx, idx2word = load_vocab()
    with codecs.open(path, 'r',encoding = 'utf-8') as f:
        embeddings = np.random.uniform(-0.5, 0.5, [len(word2idx), embedding_size])#
        # for line in f:
        #     line = line.strip().split()
        #     word = line[0]
        #     #print(word)
        #     embedding = [float(x) for x in line[1:]]
        #     if word in word2idx:
        #         embeddings[word2idx[word]] = embedding
        #         cnt += 1
        #print(cnt)
    embeddings[word2idx['<PAD>']] = np.zeros(embedding_size)
    print("oov", len(word2idx) - cnt)
    print("total_word", len(word2idx))
    print("embeddings PAD", embeddings[word2idx['<PAD>']])
    print("embeddings UNK", embeddings[word2idx['<UNK>']])
    return embeddings.astype(np.float32)

def is_number(str):
        try:
            if str=='NaN':
                return False
            float(str)
            return True
        except ValueError:
            return False

def create_data(source_sents, target_sents, key_words): 
    word2idx, idx2word = load_vocab()
    x_list, y_list, Sources, y_len = [], [], [], []
    keys = []
    for source_sent, target_sent, key in zip(source_sents, target_sents, key_words):  #for循环取出每一个句子
        print(source_sent)
        print(target_sent)
        x = [word.lower() for word in source_sent] # 1: OOV, </S>: End of Text
        # 如果指定的键值不在，则返回该默认值
        y = [word2idx.get(word, word2idx['<UNK>']) for word in (target_sent + ["</S>"])]
        ###record source order
        souce_X = []
        if max(len(x), len(y)) <= hp.maxlen:
            new_x = []
            count = 1
            temp = []
            source_X = []
            print(x)
            for word in x[1:]: #######skip first "equ" 
                if word == "equ":
                    new_x.append(temp)
                    temp = []
                    continue
                if word == ":":
                    continue
                # 如果不是结尾，暂时存入temp
                temp.append([word, count])
                source_X.append(word2idx.get(word, word2idx['<UNK>']))

                count += 1

            new_x.append(temp)

            if len(new_x) >= 3:######we dont handle more than 2 equs questions
                print("too many equs...")
                continue
            
            x_list.append(new_x)

            y_list.append(y)
            Sources.append(source_X)
            key = [word2idx.get(word.lower(),word2idx['<UNK>']) for word in  key]
            keys.append(key)
            y_len.append(len(y))

    trees = []
    X_input = []
    #print(len(x_list))
    Y_list = []
    Y_len = []

    key_out = []
    source_X_word= []
    for equ_x, input_x, a , b , key in zip(x_list, Sources, y_list, y_len, keys):
        # 如果只有一个等式
        if len(equ_x) == 1:
            equ = equ_x[0]
            source_word = [word[0] for word in equ]
            #print(sen)
            ####we dont have to add pseudo root so wo sub 1 for in every index of node
            ##we also dont need modify input_x
            for item in equ:
                item[1] = item[1] - 1
           
            try:
                root = build_tree(equ)
            except IndexError:
                continue
            if root == "wrong_tree":
                continue
            else:
                trees.append(root)
                source_X_word.append(source_word)
                assert len(source_word) == len(input_x)
                X_input.append(input_x)
                key_out.append(key)
                Y_list.append(a)
                Y_len.append(b)

        else: ###### we add a pseudo root as root
            root = Tree("pseudo_root", 0) ###### We assign 0 to pseudo root
            ### we need add preudo in input_x at first position because it index is 0 
            input_x.insert(0, word2idx["pseudo_root"])

            source_word = ["pseudo_root"] + [word[0] for word in equ_x[0]] + [word[0] for word in equ_x[1]]
            try:
                root.left = build_tree(equ_x[0])
                root.right = build_tree(equ_x[1])
            except IndexError:
                continue
            if root.left == "wrong_tree" or root.right =="wrong_tree":
                continue
            else:
                trees.append(root)
                X_input.append(input_x)

                source_X_word.append(source_word)
                assert len(source_word) == len(input_x)
                Y_list.append(a)
                Y_len.append(b)
                key_out.append(key)
    
    print("total: " + str(len(x_list) - len(trees)) + " invilid equations.")
    print(len(trees))


    print(X_input[0])
    print(y_list[0])
    assert len(trees) == len(X_input) == len(Y_list) == len(Y_len) == len(key_out)
    
    ########## add tag for x input
    tag_X_input = []
    for sent in source_X_word:
        temp = []
        for word in sent:
            if is_number(word):
                temp.append(0)
            elif word.lower() == "x" or word.lower() == "y" or word.lower() == "z":
                temp.append(1)
            else:
                temp.append(2)
        tag_X_input.append(temp)

    return trees , X_input, Y_list, Y_len, tag_X_input, key_out

def load_train_data():
    #de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    #en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    # need fix: key_words?
    data = json.load(open(hp.train, 'r'))
    source = []
    target = []
    key_words = []
    for line in data:
        if "text" not in line:
            continue
        #if len(line["best_answer"]) == 0:
            #continue
        equ = line["equations"].strip().replace("\r\n", "").split()
        equ = [word.lower() for word in equ]
        text = line["text"].strip().replace("\r\n", "").split()
        text = [word.lower() for word in text]
        #for l in line["best_answer"]:
            #answer.extend(l.strip().split())
        if len(text)) == 0:
            continue
        key_word = line["key"].strip().replace(",", "").split()

        key_word = [word.lower() for word in key_word]
        if len(key_word) == 0:
            val = min(len(text), 4)
            key_word = random.sample(text), val)
        #text = line["original_text"]
        #answer = line["best_answer"]
        #equation = line["equation"]
        #text.append("/seg")
        #text.extend(equation)

        source.append(equ))
        target.append(text)
        key_words.append(key_word)

    trees , X_input, y_list, Y_len, tag_X_input , key_out = create_data(source, target, key_words)
    return trees , X_input, y_list, Y_len, tag_X_input, key_out

def load_test_data():
       
    data = json.load(open(hp.test, 'r'))
    source = []
    target = []
    key_words = []
    count = 0

    #######150 key_word 0
    for line in data:
        
        if "text" not in line:
            continue
        text = []
        #answer = []
        equation = []
        answer = line["text"].strip().replace("\r\n", "").split()
        answer = [word.lower() for word in  answer]
        #for l in line["best_answer"]:
            #answer.extend(l.strip().split())
        text = line["equations"].strip().replace("\r\n", "").split()
        if len(answer) == 0:
            continue
        key_word = line["key"].strip().replace(",", "").split()
        if len(key_word) == 0:
            val = min(len(answer), 4)
            key_word = random.sample(answer, val)
        #text.append("/seg")
        #text.extend(equation)

        source.append(text)
        target.append(answer)
        key_words.append(key_word)

    #print(target[:10])
    trees , X_input, y_list, Y_len , tag_X_input, key_out = create_data(source, target, key_words)
    #print(len(y_list[0]))
    #print(y_list[:10])
    dev_size = int(len(X_input) * 0.9)
    for i , j in zip(y_list, Y_len):
        if len(i) != j:
            print("bug")
    dev_X_trees = trees[:dev_size]
    dev_X_input = X_input[:dev_size]
    dev_y_list = y_list[:dev_size]
    dev_Y_len = Y_len[:dev_size]
    dev_tag_X = tag_X_input[:dev_size]
    dev_key_out = key_out[:dev_size]


    test_X_trees = trees[dev_size:]
    test_X_input = X_input[dev_size:]
    test_y_list = y_list[dev_size:]
    test_Y_len = Y_len[dev_size:]
    test_tag_X = tag_X_input[dev_size:]
    test_key_out = key_out[dev_size:]
    #preorder(trees[0])
    #inorder(trees[0])
    #print(X_input[0])
    #print(y_list[0])
    #print(Y_len[0])


    return dev_X_trees,dev_X_input,dev_y_list,dev_Y_len,dev_tag_X, dev_key_out,test_X_trees,test_X_input , \
        test_y_list,test_Y_len,test_tag_X, test_key_out

if __name__ == "__main__":
    load_test_data()