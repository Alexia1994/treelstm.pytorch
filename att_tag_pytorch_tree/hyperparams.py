# -*- coding: utf-8 -*-

'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    train = 'data/key_eval.json'
    test = 'data/key_dev.json'
    # training
    batch_size = 32 # alias = N
    #lr = 0.001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    
    name = "tree_gen"
    epoch = 50

    data = "data"
    tag_embedding_dim = 10
    attn_model =  'general'  ###### 'dot' 'cacat'
    add_tag = True
    cuda = True
    # model
    maxlen = 300 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 50
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    ###seq2seq
    hidden_dim = 256
    embedding_dim = 100 ###download 300 dim

    embedding_path = "glove.6B.50d.txt"
    #learning_rate = 0.3
    
    ###
    max_word = 20000 ### vocab max num 