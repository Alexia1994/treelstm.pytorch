from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim


# IMPORT CONSTANTS
from treelstm import Constants

from treelstm import SentimentTreeLSTM
# DATA HANDLING CLASSES
from treelstm import Vocab
# DATASET CLASS FOR SICK DATASET
from treelstm import SSTDataset
# METRICS CLASS FOR EVALUATION
from treelstm import Metrics
# UTILITY FUNCTIONS
from treelstm import utils
# CONFIG PARSER
from config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from treelstm import SentimentTrainer

# MAIN BLOCK
def main():
    # export CUDA_VISIBLE_DEVICES=3
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # argument validation
    args.cuda = False
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    # some settings
    if args.fine_grain:
        args.num_classes = 5
    else:
        args.num_classes = 3


    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    # 准备目录
    vocab_file = os.path.join(args.data, 'vocab-cased.txt') # use vocab-cased
    # NO, DO NOT BUILD VOCAB,  USE OLD VOCAB

    # get vocab object from vocab file previously written
    print(vocab_file)
    vocab = Vocab(filename=vocab_file, 
                data=[Constants.PAD_WORD, Constants.UNK_WORD,
                    Constants.BOS_WORD, Constants.EOS_WORD])
    print('==> SST vocabulary size : %d ' % vocab.size())

    # let program turn off after preprocess data
    is_preprocessing_data = False 

    # train
    train_file = os.path.join(args.data, 'sst_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SSTDataset(train_dir, vocab, args.num_classes, args.fine_grain)
        torch.save(train_dataset, train_file)
        # is_preprocessing_data = True
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    
    # dev
    dev_file = os.path.join(args.data,'sst_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SSTDataset(dev_dir, vocab, args.num_classes, args.fine_grain)
        torch.save(dev_dataset, dev_file)
        # is_preprocessing_data = True
    logger.debug('==> Size of dev data   : %d ' % len(dev_dataset))

    # test
    test_file = os.path.join(args.data,'sst_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SSTDataset(test_dir, vocab, args.num_classes, args.fine_grain)
        torch.save(test_dataset, test_file)
        # is_preprocessing_data = True
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))


    # initialize model, criterion/loss_function, optimizer
    criterion = nn.NLLLoss()
    model = SentimentTreeLSTM(
                vocab.size(),
                args.input_dim,
                args.mem_dim,
                args.num_classes,
                args.freeze_embed,
                criterion,
                device,
                args.dropout,
                args.n
            )

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sst_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(
            os.path.join(args.glove, 'glove.840B.300d'))
        logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
        emb.normal_(0, 0.05)

        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                    Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        # is_preprocessing_data = True
        torch.save(emb, emb_file)
    if is_preprocessing_data:
        print ('done preprocessing data, quit program to prevent memory leak.')
        print ('please run again.')
        quit()
    # plug these into embedding matrix inside model
    # python原地操作的后缀为 _，处理高维数据时可帮助减少内存
    model.emb.weight.data.copy_(emb)

    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # create trainer object for training and testing
    trainer = SentimentTrainer(args, model, criterion, optimizer, device)

    best = -float('inf')
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        train_loss, train_pred = trainer.test(train_dataset)
        dev_loss, dev_pred = trainer.test(dev_dataset)
        #test_loss, test_pred = trainer.test(test_dataset)
        
        train_acc = metrics.sentiment_accuracy_score(train_pred, train_dataset.labels)
        dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
        #test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
        logger.info('==> Epoch {}, Train \tLoss: {} \tAccuracy: {}'.format(
            epoch, train_loss, train_acc))
        logger.info('==> Epoch {}, Dev \tLoss: {} \tAccuracy: {}'.format(
            epoch, dev_loss, dev_acc))
        #logger.info('==> Epoch {}, Test \tLoss: {}\tAccuracy: {}'.format(
            #epoch, test_loss, test_acc))

        if best < dev_acc:
            best = dev_acc
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
                'train_acc': train_acc, 'dev_acc': dev_acc,
                'args': args, 'epoch': epoch
            }
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))


if __name__ == "__main__":
    main()
