import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

class FractalExpert(nn.Module):
    def __init__(self, input_size, output_size, depth=0, max_depth=3):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(output_size, 1))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.reset_parameters()
        self.sub_experts = None
        if depth < max_depth:
            self.sub_experts = nn.ModuleList([
                FractalExpert(input_size, output_size, depth+1, max_depth) for _ in range(2)
            ])

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, depth_prob):
        if self.sub_experts and torch.rand(1).item() < depth_prob:
            return sum(e(x, depth_prob) for e in self.sub_experts) / len(self.sub_experts)
        return F.linear(x, self.weight.t(), self.bias)

class ProductKeyMemory(nn.Module):
    def __init__(self, dim, num_heads, num_keys):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_keys = num_keys
        self.keys = nn.Parameter(torch.randn(num_heads, num_keys, dim // num_heads))

    def forward(self, query):
        bsz, seqlen, _ = query.shape
        query = query.view(bsz, seqlen, self.num_heads, -1)
        scores = torch.einsum('bshd,hkd->bhsk', query, self.keys)
        return scores.view(bsz, seqlen, -1)

class FHME(nn.Module):
    def __init__(self, input_size, output_size, num_experts, num_heads, key_dim):
        super().__init__()
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.key_dim = key_dim

        num_experts_per_head = int(math.sqrt(num_experts))
        self.product_key_memory = ProductKeyMemory(key_dim, num_heads, num_experts_per_head)
        self.experts = nn.ModuleList([FractalExpert(input_size, output_size) for _ in range(num_experts)])
        self.output_proj = nn.Linear(output_size, output_size)
        self.expert_dropout = nn.Dropout(0.1)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        
        # Generate keys
        keys = self.product_key_memory(x)
        
        # Select top-k experts
        k = min(32, self.num_experts)
        top_k_scores, top_k_indices = keys.topk(k, dim=-1)
        top_k_scores = F.softmax(top_k_scores, dim=-1)
        
        # Apply experts
        expert_outputs = []
        for i in range(k):
            expert_idx = top_k_indices[:, :, i]
            expert_input = x.view(-1, x.size(-1))
            expert_output = torch.stack([self.experts[idx.item()](expert_input[j], 0.5) 
                                         for j, idx in enumerate(expert_idx.view(-1))])
            expert_outputs.append(expert_output.view(bsz, seqlen, -1))
        
        expert_outputs = torch.stack(expert_outputs, dim=-2)
        
        # Combine expert outputs
        combined_output = torch.sum(expert_outputs * top_k_scores.unsqueeze(-1), dim=-2)
        combined_output = self.expert_dropout(combined_output)
        
        return self.output_proj(combined_output)

class FHMETransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_experts, num_heads, key_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.fhme = FHME(d_model, d_model, num_experts, num_heads, key_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src = src + checkpoint(self.self_attn, src2, src2, src2, src_mask)[0]
        src = src + checkpoint(self.fhme, self.norm2(src))
        return src

class FHMELanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_experts, num_heads, key_dim):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            FHMETransformerBlock(d_model, nhead, num_experts, num_heads, key_dim)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for block in self.transformer_blocks:
            src = block(src, src_mask)
        return self.output_layer(src)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Example usage
vocab_size = 30000
d_model = 512
nhead = 8
num_layers = 6
num_experts = 1024 * 1024  # 1 million experts
num_heads = 8
key_dim = 64

model = FHMELanguageModel(vocab_size, d_model, nhead, num_layers, num_experts, num_heads, key_dim)

# Generate some dummy data
src = torch.randint(0, vocab_size, (20, 32))  # (sequence_length, batch_size)

# Forward pass
output = model(src)
print(f"Input shape: {src.shape}")
print(f"Output shape: {output.shape}")