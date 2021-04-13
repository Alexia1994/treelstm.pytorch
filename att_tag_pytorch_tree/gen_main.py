from __future__ import print_function

import os, time
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import sys
from torch.autograd import Variable
from nltk.translate.bleu_score import corpus_bleu
from hyperparams import Hyperparams as hp 
import json
from tree import cal_node
# NEURAL NETWORK MODULES/LAYERS
from model import *
# DATA HANDLING CLASSES
from tree import Tree
from data_load import load_vocab, load_train_data, load_test_data, get_word_vector

# TRAIN AND TEST HELPER FUNCTIONS
from trainer import Task_Trainer

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
final = os.path.join(hp.data, "final_result.txt")
final_writer = open(final, 'w')
#nn.parameters 
# MAIN BLOCK

def Train(epoch, optimizer, trainer, trees, X_input, y_list, Y_len, tag_X, key_word, embedding_tag , device):
    indices = torch.randperm(len(trees))
    k = 0
    loss = 0.0
    trainer.train()
    for idx in range(len(trees)):
    #for idx in range(hp.batch_size):
        tree , input_encoder, target, target_len , input_tag , keys = trees[indices[idx]],X_input[indices[idx]], \
            y_list[indices[idx]], Y_len[indices[idx]], tag_X[indices[idx]], key_word[indices[idx]]
        
        input_encoder = torch.LongTensor(input_encoder)        
        input_tag = torch.LongTensor(input_tag)        
        keys = torch.LongTensor(keys)

        if hp.cuda:
            input_encoder = input_encoder.to(device)
            keys = keys.to(device)
    
        emb1 = trainer.embedding_model(input_encoder)

        if hp.add_tag:
            if hp.cuda:
                input_tag = input_tag.to(device)
            emb2 = embedding_tag(input_tag)
            emb = torch.cat((emb1, emb2), dim = 1)
        else:
            emb = emb1
        emb_keys = trainer.embedding_model(keys)

        loss  += trainer(tree , emb , emb_keys, target, target_len, device)

               
        k += 1
        if k == hp.batch_size:
            loss = loss / hp.batch_size
            print("epoch {} batch_num {}  loss : {} ".format(epoch, idx // hp.batch_size , loss.item()))
            loss.backward()
            # 对encoder和decoder进行梯度裁剪
            _ = torch.nn.utils.clip_grad_norm_(trainer.parameters(), 5.) ###clip = 50

            optimizer.step()
            optimizer.zero_grad()
            k = 0
            loss = 0.0
def Test(epoch, trainer, trees, X_input, y_list, Y_len, tag_X, key_word, Y_ans, idx2word, embedding_tag, device,  dev = True):
   
    
    trainer.train()
    total_loss = 0.0
    total_pred = []
    for idx in range(len(trees)):
    #for idx in range(hp.batch_size):
        tree , input_encoder, target, target_len , input_tag , keys = trees[idx], X_input[idx], \
            y_list[idx], Y_len[idx], tag_X[idx], key_word[idx]
        
        input_encoder = torch.LongTensor(input_encoder)        
        input_tag = torch.LongTensor(input_tag)        
        keys = torch.LongTensor(keys)

        if hp.cuda:
            input_encoder = input_encoder.to(device)
            keys = keys.to(device)
    
        emb1 = trainer.embedding_model(input_encoder)

        if hp.add_tag:
            if hp.cuda:
                input_tag = input_tag.to(device)
            emb2 = embedding_tag(input_tag)
            emb = torch.cat((emb1, emb2), dim = 1)
        else:
            emb = emb1
        emb_keys = trainer.embedding_model(keys)

        if dev:
            loss = trainer(tree , emb , emb_keys, target, target_len, device)
            total_loss += loss.item()
        else:

            dev_pred = trainer.test(tree, emb, emb_keys, device)
            dev_pred = [idx2word[word] for word in dev_pred]
            total_pred.append(dev_pred)
    if not dev:
        #Y_ans = Y_ans[:hp.batch_size]
        score = corpus_bleu(Y_ans, total_pred)
        score_1 = corpus_bleu(Y_ans, total_pred, weights = (1, 0, 0, 0))
        score_2 = corpus_bleu(Y_ans, total_pred, weights = (0, 1, 0, 0))
        score_3 = corpus_bleu(Y_ans, total_pred, weights = (0, 0, 1, 0))
        score_4 = corpus_bleu(Y_ans, total_pred, weights = (0, 0, 0, 1))
        #max_score_1 = max(max_score_1, score_1)
        #max_score_2 = max(max_score_2, score_2)
        #max_score_3 = max(max_score_3, score_3)
        #max_score_4 = max(max_score_4, score_4)
        #if score > max_score:
            #max_score = score
        ans = []
        output_result = os.path.join(hp.data, str(epoch) + "_result.json")
        file_out = open(output_result, 'w')
        for source, target, result in zip(X_input,Y_ans, total_pred):
            my_store = {}

            my_store["text"] = " ".join([idx2word[ID] for ID in source]) + "\n"
            my_store["best_answer"] = " ".join(target) + "\n"
            my_store["pre_answer"] = " ".join(result) + "\n"

            ans.append(my_store)
        final_writer.write("epoch {} score {} score_1 {} score_2 {} score_3 {} score_4 {} ".format(epoch, score, score_1, score_2, score_3, score_4))
        json.dump(ans, file_out, indent = 4)
        file_out.close()
    if dev:
        return total_loss
def main():

    tag_to_index = {"number" : 0, "unknown_number" : 1, "symbol" : 2}

    device = torch.device("cuda:2" if
                    torch.cuda.is_available() else "cpu")
    #device = "cpu"
    hp.cuda = hp.cuda and torch.cuda.is_available()
    # args.cuda = False
    print("cuda use is ", hp.cuda)
    # torch.manual_seed(args.seed)

    if hp.cuda:
        torch.cuda.manual_seed(480)
    else:
        torch.manual_seed(480)

    #train_dir = os.path.join(hp.data,'train/')
    #dev_dir = os.path.join(hp.data,'dev/')
    #test_dir = os.path.join(args.data,'test/')

    # write unique words from all token files
    #token_files = [os.path.join(split, 'sents.toks') for split in [train_dir, dev_dir, test_dir]]
    #vocab_file = os.path.join(args.data,'vocab-cased.txt') # use vocab-cased
    # build_vocab(token_files, vocab_file) NO, DO NOT BUILD VOCAB,  USE OLD VOCAB

    # get vocab object from vocab file previously written
    word2idx, idx2word = load_vocab()
    print('==> vocabulary size : %d ' % len(word2idx))


    #X, Y = load_train_data()
    dev_X_trees,dev_X_input,dev_y_list,dev_Y_len, dev_tag_X ,dev_key_word, test_X_trees,test_X_input, \
        test_y_list,test_Y_len ,test_tag_X, test_key_word = load_test_data()

    # train

    train_X_trees,train_X_input,train_y_list,train_Y_len, train_tag_X ,train_key_word = load_train_data()
    #train_file = os.path.join(hp.data,'gen_train.pth')

    criterion = nn.NLLLoss()


    #embedding_model
    embedding_model = nn.Embedding(len(word2idx), hp.embedding_dim)


    if hp.add_tag:
        embedding_tag = nn.Embedding(len(tag_to_index), hp.tag_embedding_dim)
        if hp.cuda:
            embedding_tag = embedding_tag.to(device)

    #print(word2idx["x"])
    #print(embedding_model(Variable(torch.LongTensor([word2idx["x"]]))))
    #print(len(word2idx))
    # if hp.optim=='adam':
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    # elif hp.optim=='adagrad':
    #     # optimizer   = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    #     optimizer = optim.Adagrad([
    #             {'params': model.parameters(), 'lr': args.lr}
    #         ], lr=args.lr, weight_decay=args.wd)



   
    #if os.path.isfile(emb_file):
        #emb = torch.load(emb_file)
    #else:

        # load glove embeddings and vocab
        #glove_emb = get_word_vector(os.path.join(hp.data, hp.embedding_path), hp.embedding_dim)

        #emb = torch.Tensor(glove_emb)

        #torch.save(emb, emb_file)

        #print('done creating emb')


    # plug these into embedding matrix inside model
    #if hp.cuda:
        #emb = emb.to(device)

    # model.childsumtreelstm.emb.state_dict()['weight'].copy_(emb)
    #print(type(emb))
    #embedding_model.state_dict()['weight'].copy_(emb)

    #if hp.cuda:
        #embedding_model = embedding_model.to(device)


   
    if hp.add_tag:
        encoder_embedding_dim = hp.embedding_dim + hp.tag_embedding_dim

    # initialize encoder_model, decoder_model, criterion/loss_function, optimizer
    encoder_model = EncoderTreeLSTM(
                device, 
                encoder_embedding_dim, hp.hidden_dim  
            )

    decoder_model = Decoder(hp.attn_model, hp.embedding_dim, hp.hidden_dim, len(word2idx))
    #if hp.cuda:
        #criterion = criterion.to(device)
        #encoder_model.to(device), criterion.to(device), decoder_model.to(device)
    # create trainer object for training and testing
    trainer  = Task_Trainer(hp, encoder_model, decoder_model, embedding_model ,criterion, \
              word2idx)
    optimizer = optim.Adam(trainer.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = optim.SGD(trainer.parameters(), lr = hp.learning_rate)
    optimizer.zero_grad()

    
    if hp.cuda:
        trainer = trainer.to(device)
    mode = 'EXPERIMENT'
    
    #####Y_ans
   

    max_score , max_score_1, max_score_2, max_score_3, max_score_4 = 0.0, 0.0, 0.0, 0.0, 0.0
    Y_ans = [[idx2word[word] for word in sentence] for sentence in test_y_list]


    

    min_loss = 100000000
    min_epoch = 0
    if mode == "PRINT_TREE":
        for i in range(0, 10):
            print('_______________')
        print('break')

    elif mode == "EXPERIMENT":
        for epoch in range(hp.epoch):
            #train_X_trees,train_X_input,train_y_list,train_Y_len, train_tag_X ,train_key_word
            Train(epoch, optimizer, trainer, train_X_trees, train_X_input, train_y_list, train_Y_len, train_tag_X, train_key_word, embedding_tag, device)

            # trainer.eval()  #####现在还没用 因为没加BN
            # #with torch.no_grad():
            dev_X_trees,dev_X_input,dev_y_list,dev_Y_len, dev_tag_X ,dev_key_word, 
            test_X_trees,test_X_input, \
            test_y_list,test_Y_len ,test_tag_X, test_key_word
            #######dev
            LOSS = Test(epoch,trainer, dev_X_trees, dev_X_input, dev_y_list, dev_Y_len, dev_tag_X, dev_key_word, Y_ans, idx2word, embedding_tag, device, dev = True)
            if LOSS < min_loss:
                LOSS = min_loss
                min_epoch = epoch
                Test(epoch,trainer, test_X_trees, test_X_input, test_y_list, test_Y_len, test_tag_X, test_key_word, Y_ans, idx2word, embedding_tag, device, dev = False)

        print("min_epoch " , min_epoch)
if __name__ == "__main__":
    # log to console and file
    #logger1 = log_util.create_logger("temp_file", print_console=True)
    #logger1.info("LOG_FILE") # log using loggerba
    # attach log to stdout (print function)
    #s1 = log_util.StreamToLogger(logger1)
    #sys.stdout = s1
    print ('_________________________________start___________________________________')
    main()