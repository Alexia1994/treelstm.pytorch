import torch
import torch.nn as nn
import torch.nn.functional as F
# module for n_arytreelstm
class N_aryTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, criterion, device, n):
        super(N_aryTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.n = n
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)

        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        
        self.output_module = None
        self.criterion = criterion
        self.device = device
    
    def node_forward(self, inputs, child_c, child_h):

        child_h_linear_sum = torch.sum(self.iouh(child_h), dim=0, keepdim=True)
        
        iou = self.ioux(inputs) + child_h_linear_sum
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        
        # a little bit ugly
        # N*mem_dim -> 1*mem_dim
        f = []
        for idx in range(self.n):
            f.append(torch.sigmoid(
                torch.sum(self.fh(child_h), dim=0, keepdim=True) +
                self.fx(inputs))
            )
        f = torch.cat(f, dim=0)
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h


    def forward(self, tree, inputs):
        # need fix: 或许需要转移到device上
        loss = torch.zeros(1)
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)
        
        if tree.num_children == 0:
            tree.state = self.leaf_node_forward(inputs[tree.idx-1].unsqueeze(0))
        else:
            # cat(dim=0) 是上下合并(下一个拼接到上一个的下边)
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            # maybe need fix
            tree.state = self.node_forward(child_c, child_h)

        return tree.state
