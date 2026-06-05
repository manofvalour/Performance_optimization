import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.profiler import profile, record_function, ProfilerActivity


BATCH_SIZE =256;
SEQ_LEN = 128;
DIM = 128;
N_HEAD = 8;
N_LAYER = 8;
#VOCAB_SIZE = 65;
EPOCHES = 20;
lr = 1e-3;
DROPOUT=0.2;
NUM_EXPERT = 4
TOP_K =2
HIDDEN_LAYER = 128*4

device = 'cuda' if torch.cuda.is_available() else 'cpu';

# data importing and processing
class DataProcessing:
    def __init__(self, text):
        self.txt = text
        chars = sorted(set(text))
        self.length = len(chars)

        self.stoi = {ch:i for i, ch in enumerate(chars)};
        self.itos = {i:ch for i, ch in enumerate(chars)};

    def encode_and_split(self, text, n):
        encode = lambda s:[self.stoi[c] for c in s]

        data = torch.tensor(encode(text), dtype=torch.long);
        if n:
            size = int(n * len(data));
            train_data, val_data = data[:size], data[size:]
            return train_data, val_data, self.length
        elif n ==0.0:
            return data, self.length

    def decode(self, idx):
        return ''.join(self.itos[i] for i in idx)


class CausalMultiheadAttn(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.n_heads = num_heads
        self.head_size = head_size
        DIM_TOTAL = head_size * num_heads

       # self.attn_heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)]);
        self.qkv = nn.Linear(DIM, 3 * DIM_TOTAL, bias=False)
        self.proj = nn.Linear(DIM_TOTAL, DIM);
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.res_dropout = nn.Dropout(DROPOUT)

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                'mask', 
                torch.tril(torch.ones(SEQ_LEN, SEQ_LEN))
                .view(1,1, SEQ_LEN, SEQ_LEN));

        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        B,T,C = x.shape
        qkv = self.qkv(x)

        q,k,v = qkv.chunk(3, dim=-1)
        q = q.view(B,T,self.n_heads, self.head_size).transpose(1,2);
        k = k.view(B,T,self.n_heads, self.head_size).transpose(1,2);
        v = v.view(B,T,self.n_heads, self.head_size).transpose(1,2);
        
        if use_cache:
            if self.cache_k == None:
                self.cache_k = k
                self.cache_v = v
            
            else:
                self.cache_k = torch.cat(
                    [self.cache_k, k],
                    dim =2)

                self.cache_v = torch.cat(
                    [self.cache_v, v], 
                    dim=2)
        
        k = self.cache_k
        v = self.cache_v

        if use_cache:
            if self.flash:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=False, 
                    dropout_p=self.attn_dropout.p if self.training else 0)
            else:
                weights = (q @ k.transpose(-2,-1)) * (k.size(-1)**-0.5)
                weights = weights.masked_fill(self.mask[:, :, :T, :T]==0, float("-inf"));
                weights = self.attn_dropout(weights.softmax(dim=-1))

                out = weights @ v
        
        else:
            if self.flash:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, 
                    dropout_p=self.attn_dropout.p if self.training else 0)
            else:
                weights = (q @ k.transpose(-2,-1)) * (k.size(-1)**-0.5)
                weights = weights.masked_fill(self.mask[:, :, :T, :T]==0, float("-inf"));
                weights = self.attn_dropout(weights.softmax(dim=-1))

                out = weights @ v

        out = out.transpose(1,2).contiguous().view(B,T,C)
        out = self.res_dropout(self.proj(out))
     
        return out

class Router(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.layer = nn.Linear(dim, num_experts)

    def forward(self, x):
        weights = F.softmax(self.layer(x), dim=1)
        topk_vals, topk_idx = torch.topk(weights, self.top_k, dim=1)
        topk_vals_norm = topk_vals/topk_vals.sum(dim=1, keepdim=True)

        return weights, topk_vals_norm, topk_idx


class Expert(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.input_proj(x)
        x = self.act(x)
        x = self.output_proj(x)

        return self.dropout(x)

class MoELayer(nn.Module):
    def __init__(self, embedding_dim, ff_dim, num_experts,
                top_k, dropout = 0.1):
        super().__init__()
        assert 1<=top_k <=num_experts, "top_k must be between 1 and num_experts"

        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.num_experts = num_experts
        self.router= Router(embedding_dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(embedding_dim, ff_dim, dropout) for _ in range(num_experts)])
    
    def load_balance_loss(self, router_probs, top_k_indices):
        total_tokens, total_experts = router_probs.shape
        importance = router_probs.mean(dim =0)
        all_selected_experts = top_k_indices.reshape(-1)
        load = (
            torch.bincount(all_selected_experts, minlength=total_experts)
            .float()/(total_tokens * self.top_k)
        )

        loss = total_experts * (importance * load).sum()

        return loss

    def forward(self,x):
        B,T,C = x.shape
        x_flattened = x.reshape(B*T, C)

        router_probs, top_k_weights, top_k_indices = self.router(x_flattened)
        output = torch.zeros_like(x_flattened)

        for expert_id in range(self.num_experts):
            mask = (top_k_indices == expert_id)

            if not mask.any():
                continue

            token_ids, k_positions = mask.nonzero(as_tuple=True)

            expert_input = x_flattened[token_ids]
            expert_output = self.experts[expert_id](expert_input)
            weights = top_k_weights[token_ids, k_positions].unsqueeze(-1)

            output[token_ids] += expert_output * weights
            aux_loss = self.load_balance_loss(router_probs, top_k_indices)
            output = output.reshape(B, T, C)

            return self.dropout(output), aux_loss

#class MLP(nn.Module):
 #   def __init__(self, n_embed):
  #      super().__init__()
   #     self.net = nn.Sequential(nn.Linear(n_embed, 4*n_embed),
    #                nn.GELU(), nn.Linear(4*n_embed, n_embed),)
     #   self.dropout = nn.Dropout(DROPOUT)

 #   def forward(self, x):
  #      x = self.net(x)
   #     return self.dropout(x);

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head, hidden_layer,
                num_expert, top_k, dropout):
        super().__init__()
        head_size = n_embed//n_head
        self.attn= CausalMultiheadAttn(n_head, head_size);

        self.moe = MoELayer(n_embed, hidden_layer, num_expert,
                            top_k, dropout)

        self.ln1 = nn.LayerNorm(n_embed);
        self.ln2 = nn.LayerNorm(n_embed);

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        moe_output, aux_loss = self.moe(self.ln2(x))
        x = x+moe_output

        return x, aux_loss

class GPT(nn.Module):
    def __init__(self, vocab_size, 
                moe_aux_loss_coef=0.01):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            token_embed_table = nn.Embedding(vocab_size, DIM),
            pos_embed_table = nn.Embedding(SEQ_LEN, DIM),
            drop = nn.Dropout(DROPOUT),
            blocks = nn.ModuleList([TransformerBlock(DIM, N_HEAD, HIDDEN_LAYER, NUM_EXPERT, TOP_K, DROPOUT) for _ in range(N_LAYER)]),
            ln_f = nn.LayerNorm(DIM),
        ))
        
        self.moe_aux_loss= moe_aux_loss_coef
        self.lm_head = nn.Linear(DIM, vocab_size, bias=False)
        
        #weight tying
        self.transformer.weight = self.lm_head.weight;

        self.apply(self._init_weights)
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2*N_LAYER)**0.5)


    # weight initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, target = None):
        B,T = index.size()
        assert T<= SEQ_LEN, "Input sequence length exceed maximum sequence length"

        tok_emb = self.transformer.token_embed_table(index) 
        pos_emb = self.transformer.pos_embed_table(torch.arange(T, device =index.device))

        x = self.transformer.drop(tok_emb + pos_emb) 

        aux_losses = []
        for blocks in self.transformer.blocks:
            x, aux_l = blocks(x)
            aux_losses.append(aux_l)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            logits = logits.view(-1,logits.size(-1))
            targets = target.reshape(-1)
            loss = F.cross_entropy(logits, targets);

        aux_loss = torch.stack(aux_losses).mean() * self.moe_aux_loss

        return logits, loss, aux_loss
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= SEQ_LEN else idx[:, -SEQ_LEN:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]/temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits<v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


#loading data from memory
def load_text(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def get_batch(data, batch_size, seq_len):

    starts = torch.randint(
        0, len(data) - seq_len-1, size=(batch_size,)
    )

    X = torch.stack([data[i:i+seq_len] for i in starts])
    y = torch.stack([data[i+1:i+seq_len+1] for i in starts])

    return (X.clone().detach(),
           y.clone().detach())

@torch.no_grad()
def evaluate(model, val_data):

    model.eval()
    losses = []

    for _ in range(50):
        X, y = get_batch(val_data, BATCH_SIZE, SEQ_LEN)

        X = X.to(device)
        y = y.to(device)

        with torch.amp.autocast('cuda', dtype = torch.float16):
            _, loss, _ = model(X, y)
        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)

## Data and target
def train_loop(text_file):
    data = DataProcessing(text_file)
    train, val, _ = data.encode_and_split(text_file, 0.8);
    
    model = GPT(vocab_size=128).to(device)
    model = torch.compile(model)
    print(f"model size: {model.get_num_params()}")
    torch.set_float32_matmul_precision("high")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=True);
    scaler = torch.amp.GradScaler('cuda')
    #num_batches = len(train)//(BATCH_SIZE*SEQ_LEN)
    #print(num_batches)
    num_batches = 200

    for epoch in range(EPOCHES):
        total_loss = 0.0;
        aux_total_loss = 0.0;
        
        model.train()
        start = time.time()
        for _ in range(num_batches):

            optimizer.zero_grad(set_to_none =True)
            X,y = get_batch(train, BATCH_SIZE, SEQ_LEN)
            X = X.pin_memory().to(device,non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logit, loss, aux_loss = model(X,y)
            
            total = loss + aux_loss
            total_loss += total.item()
            aux_total_loss+=aux_loss.item()

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0)

            scaler.step(optimizer)
            scaler.update()

        avg_loss = total_loss/(num_batches)
        avg_aux_loss = aux_total_loss/(num_batches)
        end = time.time()
        time_change = end-start
        val_loss = evaluate(model, val)

        toks_per_sec = (num_batches * BATCH_SIZE * SEQ_LEN)/time_change;
        print(f"epoch: {epoch+1}/{EPOCHES} | train_loss: {avg_loss:.2f} | val_loss:{val_loss:.2f} | aux_loss: {avg_aux_loss:.2f} | epoch_time: {time_change:.2f} | toks/sec: {toks_per_sec:.0f} ")
        
        if epoch%5 ==0 and epoch !=0:
            text = "Having  this  thought  in  mind"
            text_enc,  _ = data.encode_and_split(text, 0.0)
            idx = text_enc.detach().clone().to(dtype=torch.long, device=device).unsqueeze(0)
            da = model.generate(idx, max_new_tokens=100, top_k =40)
            gen_text = data.decode(da[0].tolist())
            print(gen_text)




if __name__ == "__main__":
    text_file = load_text("cuda_kernels/transformer/wizard_of_oz.txt");
    train_loop(text_file)
