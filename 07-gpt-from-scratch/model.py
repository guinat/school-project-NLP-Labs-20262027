import torch
from torch import nn
from torch.nn import functional as F

# B: taille du batch, T: longueur du contexte, C: n_embd, hs: head_size, V: vocab_size


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # on divise par sqrt(hs), la dim sommée dans q.k, pour garder une variance de 1
        wei = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, hs)
        return wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads * hs)
        return self.dropout(self.proj(out))  # (B, T, C)


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # (B, T, 4C)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # (B, T, C)
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)  # (B, T, C)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # (B, T, C), les tokens échangent de l'information
        x = x + self.ffwd(self.ln2(x))  # (B, T, C), chaque token est traité seul
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos = torch.arange(T, device=idx.device)  # (T,)
        pos_emb = self.position_embedding_table(pos)  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C), pos_emb est broadcasté sur le batch
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)

        if targets is None:
            loss = None
        else:
            logits = logits.view(B * T, -1)  # (B*T, V)
            targets = targets.view(B * T)  # (B*T,)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]  # (B, min(T, block_size))
            logits, _ = self(idx_cond)  # (B, T, V)
            logits = logits[:, -1, :]  # (B, V)
            probs = F.softmax(logits, dim=-1)  # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
