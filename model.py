import torch.nn as nn 
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class Caus_attn_Mech (nn.Module):
  def __init__(self,in_dim :int, out_dim :int, num_head :int, context_len: int):
    super().__init__()
    assert(in_dim % num_head == 0)
    self.head_dim = in_dim// num_head
    self.num_head = num_head
    self.q_w = nn.Linear(in_dim, out_dim, bias = False)
    self.k_w = nn.Linear(in_dim, out_dim, bias = False)
    self.v_w  = nn.Linear(in_dim, out_dim, bias = False)
    self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal =1 ))
    self.drop_out = nn.Dropout(0.12)
    self.output_proj = nn.Linear(out_dim, in_dim)


  def forward(self, x :torch.Tensor) -> torch.Tensor:
      query = self.q_w(x)
      key = self.k_w(x)
      value = self.v_w(x)

      in_shape = x.shape



      query = query.contiguous().view(in_shape[0],in_shape[1], self.num_head, self.head_dim)
      key = key.contiguous().view(in_shape[0],in_shape[1], self.num_head, self.head_dim)
      value = value.contiguous().view(in_shape[0],in_shape[1], self.num_head, self.head_dim)

      query = query.transpose(1,2)#transpose axis (num_head and number of tokens)
      key = key.transpose(1,2)# to align emb_dimension with it its respective token
      value = value.transpose(1,2)

      mask_bool = self.mask.bool()[:in_shape[1], :in_shape[1]]


      attn_weights = query @ key.transpose(2,3)

      attn_weights.masked_fill_(mask_bool, -torch.inf)
      attn_weights = torch.nn.functional.softmax(attn_weights/math.sqrt(self.head_dim), dim = -1)
      attn_weights = self.drop_out(attn_weights)

      attention_scores = attn_weights @ value
      context_vec = attention_scores.transpose(1,2)
      context_vec = context_vec.contiguous().view(in_shape[0], in_shape[1], in_shape[2])

      return self.output_proj(context_vec)



class FF_Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.sequential = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class Transformer_arc(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.FF = FF_Layer(cfg)
        self.attn = Caus_attn_Mech(
            in_dim=cfg["emb_dim"],
            out_dim=cfg["emb_dim"],
            num_head=cfg["head_num"],
            context_len=cfg["context_len"]
        )
        self.layer_Norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.layer_Norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        shortcut = x
        x = self.layer_Norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = shortcut + x  # First residual connection


        shortcut = x
        x = self.layer_Norm2(x)
        x = self.FF(x)
        x = self.dropout(x)
        x = shortcut + x  # Second residual connection

        return x


class TPT_v1(nn.Module):
    def __init__(self, cfg :dict):
        super().__init__()
        self.t_layer = cfg["t_layer"]
        self.transformer_blocks = nn.ModuleList([
            Transformer_arc(cfg) for _ in range(self.t_layer)
        ])
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.layer_norm = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.output_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.layer_norm(x)
        logits = self.output_head(x)

        return logits


class gate_network (nn.Module):
  def __init__(self, num_expert: int, k: int, emb_dim: int):
    super().__init__()  # Crucial call to initialize the base nn.Module class
    self.net = nn.Sequential(
        nn.Linear(emb_dim, 4 * emb_dim),
        nn.GELU(),
        nn.Linear(4 * emb_dim, num_expert)
    )
    self.k = k
    self.num_expert = num_expert

  def forward (self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: #x:shape [batch_size * tkn_len, emb_dim]
    logits = self.net(x)
    weights = F.softmax(logits, dim=-1)
    values, k_indices  = torch.topk(weights, self.k, dim=-1)

    #
    values = values / (values.sum(dim=-1, keepdim=True) + 1e-20)

    
    empty = torch.zeros_like(weights)
    weights = torch.scatter(empty, dim=-1, index=k_indices, src=values)

    return weights, k_indices

class forward_layer (nn.Module):
  def __init__ (self, emb_dim: int):
    super().__init__()
    self.expert  = nn.Sequential(
                        nn.Linear (emb_dim, 4 * emb_dim),
                        nn.GELU(),
                        nn.Linear (4 * emb_dim, emb_dim)
                    )
    self.layernorm = nn.LayerNorm(emb_dim)
  def forward (self, x: torch.Tensor) -> torch.Tensor:
    x = self.layernorm(x)
    x = self.expert(x)
    return x

class MOE_router (nn.Module):
    def __init__ (self, seq_len: int, d_in: int, d_out: int, k: int = 2, num_expert: int = 4):
        super().__init__()
        self.num_expert = num_expert
        self.k = k
        
        self.gated_net = gate_network(num_expert, k, d_in)
        
        self.experts = nn.ModuleList([forward_layer(d_in) for _ in range(num_expert)])

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len_in, emb_dim = x.size()
        x_flat = x.view(-1, emb_dim)
        weights, token2idx = self.gated_net(x_flat)

    
        expert_in = torch.repeat_interleave(x_flat, self.k, dim=0)
        token2idx_flat = token2idx.flatten()

        weighted_output = torch.zeros_like(x_flat)

        
        source_token_indices = torch.arange(x_flat.size(0), device=x.device).repeat_interleave(self.k)

        for i in range(self.num_expert):
            matched_idx = torch.where(token2idx_flat == i)[0]
            if len(matched_idx) > 0:
                expert_input = expert_in[matched_idx]
                expert_output = self.experts[i](expert_input)

                
                parent_tokens = source_token_indices[matched_idx]

                
                routing_weights = weights[parent_tokens, i].unsqueeze(1)

                
                weighted_output.index_add_(0, parent_tokens, routing_weights * expert_output)

            return weighted_output.view(batch_size, seq_len_in, emb_dim)


def compute_load_balancing_loss(weights: torch.Tensor, k_indices: torch.Tensor, num_experts: int) -> torch.Tensor:

    N = weights.size(0)

   
    dispatched = torch.zeros(N, num_experts, device=weights.device)
    dispatched.scatter_(1, k_indices, 1.0)
    f = dispatched.mean(dim=0)
    P = weights.mean(dim=0)  # shape [num_experts]


    loss = num_experts * torch.sum(f * P)
    return loss

class MoE_Transformer_Arc(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Caus_attn_Mech(
            in_dim=cfg["emb_dim"],
            out_dim=cfg["emb_dim"],
            num_head=cfg["head_num"],
            context_len=cfg["context_len"]
        )

        self.moe_router = MOE_router(
            seq_len=cfg["context_len"],
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            k=cfg.get("k", 2),
            num_expert=cfg.get("num_experts", 4)
        )
        self.layer_Norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.layer_Norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention block
        shortcut = x
        x = self.layer_Norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = shortcut + x


        shortcut = x
        x = self.layer_Norm2(x)


        batch_size, seq_len, emb_dim = x.size()


        x_routed = self.moe_router(x)
        x_routed = self.dropout(x_routed)
        x = shortcut + x_routed

        return x

class TPT_v2(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.t_layer = cfg["t_layer"]
        self.transformer_blocks = nn.ModuleList([
            MoE_Transformer_Arc(cfg) for _ in range(self.t_layer)
        ])
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.layer_norm = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.output_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.layer_norm(x)
        logits = self.output_head(x)

        return logits