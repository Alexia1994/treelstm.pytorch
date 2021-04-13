import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryTreeLeafModule(nn.Module):
    """
  local input = nn.Identity()()
  local c = nn.Linear(self.in_dim, self.mem_dim)(input)
  local h
  if self.gate_output then
    local o = nn.Sigmoid()(nn.Linear(self.in_dim, self.mem_dim)(input))
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local leaf_module = nn.gModule({input}, {c, h})
    """
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()
        self.cuda = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.cx = nn.Linear(self.in_dim, self.mem_dim)
        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        #print(self.cuda)
        # if self.cuda != "cpu":
        #     self.cx = self.cx.cuda(self.cuda)
        #     self.ox = self.ox.cuda(self.cuda)

    def forward(self, input):
        c = self.cx(input)
        o = F.sigmoid(self.ox(input))
        h = o * F.tanh(c)
        return c, h

class BinaryTreeComposer(nn.Module):
    """
  local lc, lh = nn.Identity()(), nn.Identity()()
  local rc, rh = nn.Identity()(), nn.Identity()()
  local new_gate = function()
    return nn.CAddTable(){
      nn.Linear(self.mem_dim, self.mem_dim)(lh),
      nn.Linear(self.mem_dim, self.mem_dim)(rh)
    }
  end

  local i = nn.Sigmoid()(new_gate())    -- input gate
  local lf = nn.Sigmoid()(new_gate())   -- left forget gate
  local rf = nn.Sigmoid()(new_gate())   -- right forget gate
  local update = nn.Tanh()(new_gate())  -- memory cell update vector
  local c = nn.CAddTable(){             -- memory cell
      nn.CMulTable(){i, update},
      nn.CMulTable(){lf, lc},
      nn.CMulTable(){rf, rc}
    }

  local h
  if self.gate_output then
    local o = nn.Sigmoid()(new_gate()) -- output gate
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end
  local composer = nn.gModule(
    {lc, lh, rc, rh},
    {c, h})    
    """
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeComposer, self).__init__()
        self.cuda = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh
        def input_gate():
            return nn.Linear(self.in_dim, self.mem_dim)


        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()
        self.olh, self.orh = new_gate()

        self.i_g = input_gate()
        self.lf_g = input_gate()
        self.rf_g = input_gate()
        self.o_g = input_gate()
        self.u_g = input_gate()

        # if self.cuda != "cpu":
        #     self.ilh = self.ilh.cuda(self.cuda)
        #     self.irh = self.irh.cuda(self.cuda)
        #     self.lflh = self.lflh.cuda(self.cuda)
        #     self.lfrh = self.lfrh.cuda(self.cuda)
        #     self.rflh = self.rflh.cuda(self.cuda)
        #     self.rfrh = self.rfrh.cuda(self.cuda)
        #     self.ulh = self.ulh.cuda(self.cuda)
        #     self.urh = self.urh.cuda(self.cuda)
        #     self.olh =self.olh.cuda(self.cuda)
        #     self.orh = self.orh.cuda(self.cuda)

        #     self.i_g = self.i_g.cuda(self.cuda)
        #     self.lf_g = self.lf_g.cuda(self.cuda)
        #     self.rf_g = self.rf_g.cuda(self.cuda)
        #     self.o_g = self.o_g.cuda(self.cuda)
        #     self.u_g = self.u_g.cuda(self.cuda)

    def forward(self, input, lc, lh , rc, rh):
        i = F.sigmoid(self.ilh(lh) + self.irh(rh) + self.i_g(input))
        lf = F.sigmoid(self.lflh(lh) + self.lfrh(rh) + self.lf_g(input))
        rf = F.sigmoid(self.rflh(lh) + self.rfrh(rh) + self.rf_g(input))
        o = F.sigmoid(self.olh(lh) + self.orh(rh) + self.o_g(input))

        update = F.tanh(self.ulh(lh) + self.urh(rh) + self.u_g(input))
        c =  i* update + lf*lc + rf*rc
        h = o * F.tanh(c)
        return c, h






class BinaryTreeLSTM(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.leaf_module = BinaryTreeLeafModule(cuda,in_dim, mem_dim)
        self.composer = BinaryTreeComposer(cuda, in_dim, mem_dim)

    # def getParameters(self):
    #     """
    #     Get flatParameters
    #     note that getParameters and parameters is not equal in this case
    #     getParameters do not get parameters of output module
    #     :return: 1d tensor
    #     """
    #     params = []
    #     for m in [self.ix, self.ih, self.fx, self.fh, self.ox, self.oh, self.ux, self.uh]:
    #         # we do not get param of output module
    #         l = list(m.parameters())
    #         params.extend(l)

    #     one_dim = [p.view(p.numel()) for p in params]
    #     params = F.torch.cat(one_dim)
    #     return params

    def forward(self, tree, embs):
        # add singleton dimension for future call to node_forward
        # embs = F.torch.unsqueeze(self.emb(inputs),1)

        if tree.left == None:
            # leaf case
            tree.state = self.leaf_module.forward(embs[tree.index])
        else:
            
            _ = self.forward(tree.left, embs)
            _ = self.forward(tree.right, embs)    
            lc, lh, rc, rh = self.get_child_state(tree)
            tree.state = self.composer.forward(embs[tree.index],lc, lh, rc, rh)

        return tree.state


    def get_child_state(self, tree):
        lc, lh = tree.left.state
        rc, rh = tree.right.state
        return lc, lh, rc, rh

######encoder tree-lstm   
class EncoderTreeLSTM(nn.Module):
    def __init__(self, cuda,  in_dim, mem_dim ):
        super(EncoderTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.tree_module = BinaryTreeLSTM(cuda, in_dim, mem_dim)


    def get_state(self, tree):
        ############这里返回的长度可能不等于输入长度 因为构造树的时候丢掉了 括号 
        if tree.left != None:
            L = torch.cat((tree.state[1].unsqueeze(0), self.get_state(tree.left)), dim = 0)
            return torch.cat((L, self.get_state(tree.right)), dim = 0) 
        else:
            return tree.state[1].unsqueeze(0)
    def forward(self, tree, inputs):
        """
        TreeLSTMSentiment forward function
        :param tree:
        :param inputs: (sentence_length, 1, 300)
        :param training:
        :return:
        """
        tree_state  = self.tree_module(tree, inputs)
        
        return tree_state

###Decoder LSTM
# need fix: no beam search?
class Decoder(nn.Module):
    def __init__(self, attn_model,embedding_size, hidden_size, output_size, dropout=0.1):
        super(Decoder, self).__init__()

        # 保存到self里，attn_model就是前面定义的Attn类的对象。
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size  #####vocab_size
        
        self.dropout = dropout

        # 定义Decoder的layers
      

        ##self.embedding_dropout = nn.Dropout(dropout)

        ####
        #print(embedding_size)
        #print(hidden_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first = True)
        #self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)


        self.attn = Attn(self.attn_model, hidden_size)

        self.concat = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, input_step, last_hidden ,embedding, encoder_tree_outputs):
        # 注意：decoder每一步只能处理一个时刻的数据，因为t时刻计算完了才能计算t+1时刻。
        # input_step的shape是(1, 1)，1是batch，1是当前输入的词ID(来自上一个时刻的输出)
        # 通过embedding层变成(1, 1, 500)，然后进行dropout，shape不变。


        ###input_step [1,1]
        embedded = embedding(input_step)
        #print(embedded.size())
        #print(last_hidden)
        #embedded = self.embedding_dropout(embedded)
        # 把embedded传入GRU进行forward计算
        # 得到rnn_output的shape是(1, 64, 500)
       
        rnn_output, hidden = self.lstm(embedded, last_hidden)
        # 计算注意力权重， 
        

        attn_weights_tree = self.attn(rnn_output, encoder_tree_outputs)
        
        #attn_weights_key = self.attn(rnn_output, encoder_key_outputs)

        # encoder_outputs是(scr_len, hidden_size) 

        # attn_weights [src_len]
        
      

        #print("attn_weights", attn_weights.shape)
        #print("encoder_outputs", encoder_outputs.shape)

        attn_weights_tree = attn_weights_tree.unsqueeze(0)

        #attn_weights_key = attn_weights_key.unsqueeze(0)

        context_tree = torch.mm(attn_weights_tree, encoder_tree_outputs)

        #context_key = torch.mm(attn_weights_key, encoder_key_outputs)


        #print("context", context.shape) ####[1, hiddensize]

        


        #print("rnn_output", rnn_output.shape)  ########由于计算atten 时候改变了形状 但是呢还是没变[1,1,hiddensize]
        
        rnn_output = rnn_output.squeeze(0)
        concat_input = torch.cat((rnn_output, context_tree), 1)
      
        #concat_input = torch.cat((concat_input, context_tree), 1)
        #print("concat_input", concat_input.shape)
        #[1, 2 * hidden]

        #concat_output = torch.tanh(self.concat(concat_input))  
        #####要不要加激活函数

        concat_output = self.concat(concat_input)
 
       
        
        #########
        
        #print(hidden[0])
        
        #rnn_output = rnn_output.squeeze(0).squeeze(0)
        #print(rnn_output)

        concat_output = concat_output.squeeze(0)

        #print("concat_output", concat_output.shape)
        output = self.out(concat_output)
        probability = F.softmax(output)
        #print(type(probability))
        # 用softmax变成概率，表示当前时刻输出每个词的概率。
        output = F.log_softmax(output)
        #print("out", output.size())
        # 返回 output和新的隐状态

        return output, probability, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        # elif self.method == 'concat':
        #     self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
        #     self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        # 输入hidden的shape是( 1 , hidden_size)
        # encoder_outputs的shape是(src_len, hidden_size)
        # hidden * encoder_output得到的shape是(500)，然后对第3维求和就可以计算出score。
        return torch.sum(hidden * encoder_output, dim= 1 )

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=1)

    # def concat_score(self, hidden, encoder_output):
    #     energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
    #     return torch.sum(self.v * energy, dim=2)
    
    # 输入是上一个时刻的隐状态hidden和所有时刻的Encoder的输出encoder_outputs
    # 输出是注意力的概率，也就是长度为input_lengths的向量，它的和加起来是1。
    def forward(self, hidden, encoder_outputs):
        # 计算注意力的score，输入hidden的shape是(1, 1,hidden_size),表示t时刻数据的隐状态

        # encoder_outputs的shape是(src_len ,hidden_size)
        hidden = hidden.squeeze(0) ##### 我们要（1,hidden_size)

        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        #elif self.method == 'concat':
            #attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            # 计算内积，参考dot_score函数
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        # 把attn_energies从(max_length=10)
        

        # 使用softmax函数把score变成概率，shape仍然是(64, 10)，然后用unsqueeze(1)变成
        
        return F.softmax(attn_energies)

