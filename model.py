import torch.nn as nn 
import torch
import torch.nn.functional as F


class causal_attn (nn.Module):
    def __init__ (self,emb_dim: int, context_len: int, num_head :int  ):
        super().__init()
        
        assert(emb_dim // num_head)
        self.num_head = num_head
        self.context_len = context_len
        q_weight = nn.Linear(emb_dim, emb_dim, bias = False)
        k_weight = nn.Linear(emb_dim, emb_dim, bias = False)
        v_weight = nn.Linear(emb_dim, emb_dim, bias = False)

        self.register_buffer ('mask', torch.tril(torch.ones(self.context_len, self.context_len)))


    def forward(self, X:torch.Tensor) ->torch.Tensor: #X:shape [batcH_size, num_token, emb_dim]
        query = q_weight(x)
        key = k_weight(x)
        value = v_weight(x)

