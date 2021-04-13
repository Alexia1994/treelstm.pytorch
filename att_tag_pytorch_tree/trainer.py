from tqdm import tqdm
import torch
from torch.autograd import Variable as Var
import torch.nn.functional as F
import torch.nn as nn
from sys import exit
class Task_Trainer(nn.Module):
    """
    For Sentiment module
    """
    def __init__(self, args, encoder_model, decoder_model, embedding_model ,criterion, word2idx):
        super(Task_Trainer, self).__init__()
        self.args = args
        self.encoder_model  = encoder_model
        self.decoder_model = decoder_model
        self.embedding_model = embedding_model
        self.criterion  = criterion
        self.rnn = nn.LSTM(args.embedding_dim, args.hidden_dim, batch_first = True)
        self.word2idx = word2idx
        self.encoder_model.train()
        self.decoder_model.train()
        self.embedding_model.train()
        self.embedding_model.zero_grad()

    def forward(self, tree , emb , key_words_emb,  target, target_len, device):
        loss = 0.0
       
        target = torch.LongTensor(target)
        #print(target.size())
        #print(input_encoder)
        if self.args.cuda:
            target = target.to(device)
        #key_words_emb = F.torch.unsqueeze(key_words_emb , 0)
        #outputs, (hidden, cell) = self.rnn(key_words_emb)

        #encoder_key_outputs = outputs.squeeze(0)
        #print(encoder_key_outputs.shape)
        #return
        #emb = F.torch.unsqueeze(self.embedding_model(input_encoder), 0)
        #print(hidden.shape) [1  1 100]
        #print(cell.shape) [1 1 100]        
        #print(type(emb))
        #print(emb.size())



        ########## c0, h0       bugggggggggggggg!!!!!!!
        c0, h0 = self.encoder_model.forward(tree, emb)
        encoder_tree_outputs = self.encoder_model.get_state(tree)


        #print(encoder_outputs.shape)
        h0 = F.torch.unsqueeze(F.torch.unsqueeze(h0, 0), 0)
        c0 = F.torch.unsqueeze(F.torch.unsqueeze(c0, 0), 0)
        #params = self.model.childsumtreelstm.getParameters()
        # params_norm = params.norm()
        #print(output)
        #print(type(output))
        decoder_input = torch.LongTensor([[self.word2idx["<S>"]]])  ###add first token "S"
        if self.args.cuda:
            decoder_input = decoder_input.to(device)

        decoder_hidden = (h0 , c0)
        #print(target_len)
        for t in range(target_len):
            decoder_output, probability, decoder_hidden = self.decoder_model(
            decoder_input, decoder_hidden, self.embedding_model, encoder_tree_outputs
        )
            decoder_input = target[t].view(1,-1)
            #print(decoder_input.size())
            #print(target[t].view(1))

            #####  decoder_output has been loged
            loss += self.criterion(decoder_output.view(1,-1), target[t].view(1))


        return loss / target_len
       
        #print(loss)

       
            
             

    # helper function for testing
    def test(self, tree , emb, key_words_emb, device, max_len = 100):

        y_pred = []


        #key_words_emb = F.torch.unsqueeze(key_words_emb , 0)
        #outputs, (hidden, cell) = self.rnn(key_words_emb)
        #print(target.size())
        #print(input_encoder)
        #encoder_key_outputs = outputs.squeeze(0)


        #predictions = torch.zeros(len(dataset))
        #predictions = predictions
     
        #print(type(emb))
        #print(emb.size())
        c0, h0 = self.encoder_model.forward(tree, emb)

        h0 = F.torch.unsqueeze(F.torch.unsqueeze(h0, 0), 0)
        c0 = F.torch.unsqueeze(F.torch.unsqueeze(c0, 0), 0)


        encoder_tree_outputs = self.encoder_model.get_state(tree)
        #params = self.model.childsumtreelstm.getParameters()
        # params_norm = params.norm()
        #print(output)
        #print(type(output))
        decoder_input = torch.LongTensor([[self.word2idx["<S>"]]])  ###add first token "S"
        if self.args.cuda:
            decoder_input = decoder_input.to(device)

        decoder_hidden = (h0, c0)
        
        for t in range(100):
            decoder_output,probability, decoder_hidden = self.decoder_model(
            decoder_input, decoder_hidden, self.embedding_model, encoder_tree_outputs
        )
        # Teacher forcing: 下一个时刻的输入是当前正确答案
            decoder_input = probability.max(0)[1].view(1,-1) ##### 1是对应的索引 0 才是值 所以我们要1
            #print(decoder_input.device)
            out_index = probability.max(0)[1].cpu().numpy().tolist()
            if out_index == self.word2idx["</S>"]:
                y_pred.append(probability.max(0)[1].cpu().numpy().tolist())
                break
            y_pred.append(out_index)
            #print(decoder_input.size())
            #print(target[t].view(1))
        return y_pred


