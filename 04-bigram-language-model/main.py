import math
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from torch.nn import functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path(__file__).parent / "data" / "input.txt"

SEED = 1337
BATCH_SIZE = 4
BLOCK_SIZE = 8
LEARNING_RATE = 1e-3
EVAL_STEPS = [100, 1000, 2000, 4000, 6000, 8000, 10000]
SAMPLE_STEPS = [100, 1000, 10000]
EVAL_ITERS = 200
MAX_NEW_TOKENS = 300


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T) -> (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # (B*T, C)
            targets = targets.view(B * T)  # (B*T,)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, generator=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # (B, T, C)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, 1, generator=generator)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx


def load_text():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, encoding="utf-8") as f:
        return f.read()


def get_batch(split, splits):
    source = splits[split]
    starts = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([source[i : i + BLOCK_SIZE] for i in starts])
    y = torch.stack([source[i + 1 : i + BLOCK_SIZE + 1] for i in starts])
    return x, y  # (B, T), (B, T)


@torch.no_grad()
def estimate_loss(model, splits):
    model.eval()
    out = {}
    for split in splits:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split, splits)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, optimizer, splits, steps):
    for _ in range(steps):
        xb, yb = get_batch("train", splits)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return loss.item()


def count_bigram_loss(splits, vocab_size):
    train_data = splits["train"]
    pairs = train_data[:-1] * vocab_size + train_data[1:]
    # Lissage add-one, sinon les paires de val jamais vues en train ont une proba de 0.
    counts = torch.bincount(pairs, minlength=vocab_size**2) + 1
    probs = counts.view(vocab_size, vocab_size).float()
    probs = probs / probs.sum(1, keepdim=True)
    return {
        split: -probs[d[:-1], d[1:]].log().mean().item() for split, d in splits.items()
    }


def sample(model, decode):
    # Même seed et même token de départ, seuls les poids changent entre les samples.
    generator = torch.Generator().manual_seed(SEED)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, MAX_NEW_TOKENS, generator=generator)
    return decode(out[0].tolist())


def main():
    torch.manual_seed(SEED)
    # Tenseurs minuscules, donc des threads en plus coûtent plus qu'ils ne rapportent.
    torch.set_num_threads(1)

    text = load_text()
    print(f"corpus: {DATA_PATH}")
    n_lines = text.count("\n")
    print(f"length: {len(text):,} characters, {n_lines:,} lines")
    print(text[:1000])

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    message = "hii there"
    print(f"vocab size: {vocab_size}")
    print(f"vocab: {''.join(chars)!r}")
    print(f"encode({message!r}): {encode(message)}")
    print(f"decode(encode({message!r})): {decode(encode(message))!r}")
    print(f"round trip on message: {decode(encode(message)) == message}")
    print(f"round trip on full text: {decode(encode(text)) == text}")

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    splits = {"train": train_data, "val": val_data}
    print(f"\ndata: {tuple(data.shape)}")
    print(f"train: {len(train_data):,} tokens, val: {len(val_data):,} tokens")
    print(f"first {BLOCK_SIZE + 1} tokens: {data[: BLOCK_SIZE + 1].tolist()}")

    x = train_data[:BLOCK_SIZE]
    y = train_data[1 : BLOCK_SIZE + 1]
    for t in range(BLOCK_SIZE):
        context = x[: t + 1]
        target = y[t]
        print(
            f"t={t}: {context.tolist()} -> {target.item()}"
            f"   {decode(context.tolist())!r} -> {itos[target.item()]!r}"
        )

    xb, yb = get_batch("train", splits)
    print(f"\nxb: {tuple(xb.shape)}, yb: {tuple(yb.shape)}")
    print("inputs:")
    print(xb)
    print("targets:")
    print(yb)

    model = BigramLanguageModel(vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nparameters: {n_params:,} ({vocab_size} x {vocab_size})")
    logits, _ = model(xb)
    print(f"logits without targets: {tuple(logits.shape)}")
    logits, loss = model(xb, yb)
    print(f"logits with targets (flattened): {tuple(logits.shape)}")
    print(f"initial loss on this batch: {loss.item():.4f}")
    print(f"-ln(1/{vocab_size}): {-math.log(1 / vocab_size):.4f}")
    losses = estimate_loss(model, splits)
    print(f"step 0: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    print("\nsample before training:")
    print(sample(model, decode))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    done = 0
    for total in EVAL_STEPS:
        last_loss = train(model, optimizer, splits, total - done)
        done = total
        losses = estimate_loss(model, splits)
        print(
            f"\nstep {total}: last batch loss {last_loss:.4f}, "
            f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if total in SAMPLE_STEPS:
            print(f"sample after {total} steps:")
            print(sample(model, decode))

    losses = count_bigram_loss(splits, vocab_size)
    print(
        f"\ncount-based bigram (add-one smoothing): "
        f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    )


if __name__ == "__main__":
    main()
