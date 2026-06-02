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
lr = 1e-2;
DROPOUT=0.1;

device = 'cuda' if torch.cuda.is_available() else 'cpu';

# data importing and processing
class DataProcessing:
    def __init__(self, text):
        self.txt = text
        chars = sorted(set(text))
        self.length = len(chars)

        self.stoi = {ch:i for i, ch in enumerate(chars)};
        self.itos = {i:ch for i, ch in enumerate(chars)};

    def encode_and_split(self, n):
        encode = lambda s:[self.stoi[c] for c in s]

        data = torch.tensor(encode(self.txt), dtype=torch.long);
        size = int(n * len(data));
        train_data, val_data = data[:size], data[size:]

        return train_data, val_data, self.length

    def decode(self, idx):
        return ''.join(self.itos[i] for i in idx)

# developing the transformer architecture
#class AttentionHead(nn.Module):
 #   def __init__(self, head_size):
  #      super().__init__()
        #self.key = nn.Linear(DIM, head_size, bias=False);
        #self.query = nn.Linear(DIM, head_size, bias = False);
        #self.value = nn.Linear(DIM, head_size, bias =False);
   #     self.qkv = nn.Linear(DIM, 3*head_size, bias=False);
    #    self.dropout = nn.Dropout(DROPOUT)

     #   self.register_buffer('mask', torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)));

#    def forward(self, x):
 #       B,T,C = x.shape
       # key = self.key(x);
        #query = self.query(x);
       # value = self.value(x)

  #      qkv = self.qkv(x);

#        query, key, value = qkv.chunk(3, dim=-1)

       # weights = (query @ key.transpose(-2,-1)) * (key.size(-1)**-0.5)
        #weights = weights.masked_fill(self.mask[:T, :T]==0, float("-inf"));

      #  weights = self.dropout(weights.softmax(dim=-1))
        
 #       out = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0.0)
  #      out = self.dropout(out)
   #     return out #weights@value;

class MultiheadAttn(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.n_heads = num_heads
        self.head_size = head_size
        DIM_TOTAL = head_size * num_heads

       # self.attn_heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)]);
        self.qkv = nn.Linear(DIM, 3 * DIM_TOTAL, bias=False)
        self.proj = nn.Linear(DIM_TOTAL, DIM);
        self.dropout = nn.Dropout(DROPOUT)
        self.register_buffer('mask', torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)));


    def forward(self, x):
        B,T,C = x.shape
        qkv = self.qkv(x)

        q,k,v = qkv.chunk(3, dim=-1)
        q = q.view(B,T,self.n_heads, self.head_size).transpose(1,2);
        k = k.view(B,T,self.n_heads, self.head_size).transpose(1,2);
        v = v.view(B,T,self.n_heads, self.head_size).transpose(1,2);

       # weights = (q @ k.transpose(-2,-1)) * (k.size(-1)**-0.5)
       # weights = weights.masked_fill(self.mask[:T, :T]==0, float("-inf"));

        #weights = self.dropout(weights.softmax(dim=-1))

        #out = weights @ v
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        out = out.transpose(1,2).contiguous().view(B,T,-1)
        out = self.dropout(self.proj(out))
      #  out = out.transpose(1,2).contiguous().view(B,T, -1)
        return out #self.dropout(self.proj(out))

class MLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4*n_embed),
                    nn.GELU(), nn.Linear(4*n_embed, n_embed),)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.net(x)
        return self.dropout(x);

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.attn= MultiheadAttn(n_head, head_size);

        self.ff = MLP(n_embed)
        self.ln1 = nn.LayerNorm(n_embed);
        self.ln2 = nn.LayerNorm(n_embed);

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, DIM);
        self.pos_embed_table = nn.Embedding(SEQ_LEN, DIM)
        self.blocks = nn.Sequential(*(TransformerBlock(DIM, N_HEAD) for _ in range(N_LAYER)))
        self.ln_f = nn.LayerNorm(DIM);
        self.lm_head = nn.Linear(DIM, vocab_size)


    def forward(self, index, target = None):
        T = index.size(1)
        tok_emb = self.token_embed_table(index) 
        pos_emb =  self.pos_embed_table(torch.arange(T, device =index.device))

        x = tok_emb+pos_emb;

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            logits = logits.view(-1,logits.size(-1))
            targets = target.reshape(-1)
            loss = F.cross_entropy(logits, targets);

        return logits, loss


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
            _, loss = model(X, y)
        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)

## Data and target
def train_loop(train_data, val_data, vocab_size):
    model = GPT(vocab_size).to(device)
    model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=True);
    scaler = torch.amp.GradScaler('cuda')
    num_batches = len(train_data)//(BATCH_SIZE*SEQ_LEN)

    for epoch in range(EPOCHES):
        total_loss = 0.0;
        
        model.train()
        start = time.time()
        for _ in range(num_batches):

            optimizer.zero_grad(set_to_none =True)
            X,y = get_batch(train_data, BATCH_SIZE, SEQ_LEN)
            X = X.to(device)
            y = y.to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logit, loss = model(X,y)
            
            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()

        avg_loss = total_loss/(num_batches)
        end = time.time()
        time_change = end-start
        val_loss = evaluate(model, val_data)
        toks_per_sec = (num_batches * BATCH_SIZE * SEQ_LEN)/time_change;
        print(f"epoch: {epoch+1}/{EPOCHES}, train_loss: {avg_loss:.2f}, val_loss = {val_loss:.2f}, epoch_time: {time_change:.2f}, toks/sec: {toks_per_sec:.0f} ")


if __name__ == "__main__":
    text_file = load_text("cuda_kernels/transformer/wizard_of_oz.txt");
    data = DataProcessing(text_file)
    train, val, VOCAB_SIZE = data.encode_and_split(0.8);
    train_loop(train, val, vocab_size = 128)