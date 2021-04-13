import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hid_mem):
        super(Attention, self).__init__()

        self.hid_mem = hid_mem
        self.attn = nn.Linear(self.hid_mem * 2, hid_mem)
        # 作为nn.Module中的可训练参数使用，requires_grad属性的默认值是True
        self.v = nn.Parameter(torch.rand(hid_mem))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def score(self, hid, en_outputs):
        # en_outputs = [batch size, timestep, hid dim]
        # torch.cat(, dim), 按照维度2拼接
        # 这里采用的是Addictive Attention
        # [batch size, time step, 2*hid mem]->[batch size, time step, hid mem]
        energy = F.relu(self.attn(torch.cat([hid, en_outputs], 2)))
        # [batch size, hid mem, time step]
        energy = energy.transpose(1, 2)
        # [Batch size, 1, hid mem]
        v = self.v.repeat(en_outputs.size(0), 1).unsqueeze(1)
        # bmm: batch matrix-matrix product, [b, n, m]*[b, m, p]——>[b, n, p]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

    def forward(self, hid, en_outputs):
        # 得到encoder一共有多少个输入
        # en_outputs = [timestep, batch size, hid dim]
        timestep = en_outputs.size(0)
        # repeat从最外侧参数开始，拷贝多少个列，再拷贝多少个行
        # transpose(0, 1) 求转置
        # [time step, batch size, hid dim] ——> [batch size, time step, hid dim]
        hid = hid.repeat(timestep, 1, 1).transpose(0, 1)
        # [batch size, timestep, hid dim]
        en_outputs = en_outputs.transpose(0, 1)
        attn_energies = self.score(hid, en_outputs)
        # [batch size, 1, timestep]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # One thing to note is that the dropout argument to the LSTM is how much dropout to apply
        # BETWEEN the layers of a multi-layer RNN,
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout = dropout)
        self.attention = Attention(hid_dim)

        self.fc_out = nn.Linear(hid_dim, output_dim)
        # During training, randomly zeroes some of the elements of the input tensor with probability p.
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, en_outputs):
        
        #input = [batch size, input_dim]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]

        #input = [1, batch size]
        #en_outputs = [timestep, batch size, hid dim]

        print("input shape : " + str(input.shape))
        input = input.unsqueeze(0)
        #embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))
        # attn_weights = [batch size, 1, timestep]
        attn_weights = self.attention(hidden[-1], en_outputs)
        context = attn_weights.bmm(en_outputs.transpose(0, 1))
        # context = [1, batch size, hid dim]
        context = context.transpose(0, 1)
        # context vector和输入进行concat
        rnn_input = torch.cat([embedded, context], 2)

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]    
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        #prediction = [batch size, output dim]
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        #need fix: urequire_grad or not?
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size, require_grad=True).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            # 返回每一行最大的列标号
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs