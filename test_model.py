import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import pdb

"""RNN"""

# https://github.com/salesforce/awd-lstm-lm/blob/dfd3cb0235d2caf2847a4d53e1cbd495b781b5d2/locked_dropout.py#L5
class LockedDropout(nn.Module):
    def __init__(self, args, dropout=0.5):
        super().__init__()
        self.args = args.copy()
        self.dropout = dropout
        
    def reinit(self,x):
        self.masks = [Variable(x.data.new(x.size(1),self.args['hidden_size']).bernoulli_(1-self.dropout), requires_grad=False)
                      /(1-self.dropout) for l in range(self.args['depth'])]
        self.input = Variable(x.data.new(x.size(1),x.size(2)).bernoulli_(1-self.dropout), requires_grad=False)/(1-self.dropout)

    def forward(self, x, input_type):
        if not self.training or not self.dropout:
            return x
        if input_type=='input': return self.input*x
        else: return self.masks[int(input_type)] * x
    
#https://github.com/pytorch/pytorch/issues/11335
class LayerNormLSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)
        
        for name, param in self.named_parameters():
            # keras defaults for LSTM
            if 'weight.ih' in name:
                for gateind in range(4):
                    torch.nn.init.xavier_uniform_(param.data[hidden_size*gateind:hidden_size*(1+gateind)])
            if 'weight_hh' in name:
                #4*hidden_size, hidden_size
                for gateind in range(4):
                    torch.nn.init.orthogonal_(param.data[hidden_size*gateind:hidden_size*(1+gateind)])
                    

    def forward(self, input, hidden=None):
        self.check_forward_input(input) #size checks
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            print('LSTM cell not given hidden state')
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]') #size checks
        self.check_forward_hidden(input, cx, '[1]') #size checks
        
        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                 + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()
        
        return hy, cy
    
class pytorchLSTM(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args.copy()

        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(self.args['d'] if layer == 0 else self.args['hidden_size']),
                              hidden_size=self.args['hidden_size'])
            for layer in range(self.args['depth'])
        ])
        
        self.fc = nn.Linear(self.args['hidden_size'], self.args['num_classes'])

        self.lockdrop = LockedDropout(args,dropout=self.args['dropout'])
        
    def forward(self, input, hidden=None):
        
        input = input.transpose(0,1) #orig: batch_size, T, d
        self.lockdrop.reinit(input)
        self.args['T'], self.args['batch_size'], _ = input.size()  # supports TxNxH only
        
        if hidden is None:
            hx = input.new_zeros(self.args['depth'], self.args['batch_size'], self.args['hidden_size'], requires_grad=False)
            cx = input.new_zeros(self.args['depth'], self.args['batch_size'], self.args['hidden_size'], requires_grad=False)
            #this keeps dtypes and device same between input and hx/cx
        else:
            hx, cx = hidden

        ht = [[None, ] * (self.args['depth'])] * self.args['T']
        ct = [[None, ] * (self.args['depth'])] * self.args['T']

        h, c = hx, cx
        output = []
        for t, x in enumerate(input):
            x = self.lockdrop(x,'input')
            for l, layer in enumerate(self.hidden0):
                ht[t][l], ct[t][l] = layer(x, (h[l], c[l]))
                x = self.lockdrop(ht[t][l],str(l))
            h, c = ht[t], ct[t]
            output.append(x.clone())
        
        output = torch.stack([x for x in output]) #shape T, batch_size, hidden_size
        output = output.transpose(0,1)
        output = self.fc(output.contiguous().view(-1,self.args['hidden_size']))
        return output.view(self.args['batch_size'],self.args['T'],-1)

"""Transformer"""
# http://nlp.seas.harvard.edu/2018/04/03/attention.html

import copy, math

class transformer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args.copy()
        self.position = PositionalEncoding(d_model = self.args['d']
                                           , dropout = self.args['dropout'])
        c = copy.deepcopy
        attn = MultiHeadedAttention(h = self.args['heads']
                                    , d_model = self.args['d']
                                    , dropout = self.args['dropout'])
        ff = PositionwiseFeedForward(d_model = self.args['d']
                                     , d_ff = self.args['hidden_size']
                                    , dropout = self.args['dropout'])
        self.encoder = Encoder(EncoderLayer(self.args['d'], c(attn), c(ff)
                                            , dropout=self.args['dropout'])
                                ,N = self.args['depth'])
        self.last = nn.Linear(in_features = self.args['d'], out_features = self.args['num_classes'])
        
    def forward(self,x):
#         print(x.shape) #batch_size, T, d
        mask = Variable(subsequent_mask(self.args['T']).type_as(x))
        x = self.position(x*math.sqrt(self.args['d']))
        x = self.encoder(x, mask)
        return self.last(x)
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
                
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        position = position.type(torch.FloatTensor) 
        div_term_sin = 1 / (10000 ** (2*torch.arange(0., d_model, 2) / d_model))
        div_term_cos = 1 / (10000 ** (2*torch.arange(1., d_model, 2) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term_sin)
        pe[:, 1::2] = torch.cos(position * div_term_cos)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    "Returns a (1,size,size) matrix where everything on the diagonal and below are 1's and everything above the diagonal is 0"
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None 
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
            # bs, h, T, d/h
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        # x (bs, h, T, d/h), attn (bs, h, T, T)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    #bs, h, T, T
    p_attn = F.softmax(scores, dim = -1)
    #bs, h, T, T
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    

"""GCN"""
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
#         support = torch.mm(input, self.weight)
#         output = torch.spmm(adj, support)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' 
    
class gcn_lstm(nn.Module):

    def __init__(self, args):
        super(gcn_lstm, self).__init__()
        
        self.args = args.copy()
        self.gc1 = GraphConvolution(2*args['sig_T']+1,args['hidden_size_gcn'])
        self.gc2 = GraphConvolution(args['hidden_size_gcn'], args['embedding_size'])
        self.args['d'] = self.args['d'] + self.args['embedding_size']
        self.lstm = pytorchLSTM(self.args)
        
        
    def forward(self, X, S, A, seqlen):       
        embedding = F.relu(self.gc1(S, A))
        embedding = F.dropout(embedding, self.args['dropout'], training=self.training)
        embedding = self.gc2(embedding, A) # (sum(T_i), N_max, embedding_size)

        embedding = embedding[:, 0, :]
        
        i_list = [int(sum(seqlen[:ind].cpu().numpy())) for ind in range(len(seqlen))]
        j_list = [int(i_list[ind]+seqlen[ind].cpu().numpy()) for ind in range(len(seqlen))]
        embedding = [ torch.cat((embedding[i:j,:],embedding.new_zeros((X.shape[1]-k,self.args['embedding_size']))))  for i,j,k in zip(i_list, j_list, seqlen.cpu().numpy().astype('int'))]
        embedding = torch.stack(embedding, dim=0)
    
        X = torch.cat((X, embedding), dim=2)
        output = self.lstm(X)
        return output        

class simple_gcn_lstm(nn.Module):

    def __init__(self, args):
        super(simple_gcn_lstm, self).__init__()
        
        self.args = args.copy()
        self.gc1 = GraphConvolution(2*args['sig_T']+1,args['hidden_size_gcn'])
        self.gc2 = GraphConvolution(args['hidden_size_gcn'], args['embedding_size'])
        self.args['d'] = self.args['embedding_size']
        self.lstm = pytorchLSTM(self.args)
        
        
    def forward(self, S, A, seqlen, T):       
        embedding = F.relu(self.gc1(S, A))
        embedding = F.dropout(embedding, self.args['dropout'], training=self.training)
        embedding = self.gc2(embedding, A) # (sum(T_i), N_max, embedding_size)

        embedding = embedding[:, 0, :]
        
        i_list = [int(sum(seqlen[:ind].cpu().numpy())) for ind in range(len(seqlen))]
        j_list = [int(i_list[ind]+seqlen[ind].cpu().numpy()) for ind in range(len(seqlen))]
        embedding = [ torch.cat((embedding[i:j,:],embedding.new_zeros((T-k,self.args['embedding_size']))))  for i,j,k in zip(i_list, j_list, seqlen.cpu().numpy().astype('int'))]
        embedding = torch.stack(embedding, dim=0)
    
        output = self.lstm(embedding)
        return output        
