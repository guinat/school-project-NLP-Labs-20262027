import urllib.request
from pathlib import Path

import torch

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_PATH = Path(__file__).parent / "data" / "input.txt"
TRAIN_FRACTION = 0.9


def load_text():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(s, stoi):
    return [stoi[c] for c in s]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def load_splits():
    text = load_text()
    stoi, itos = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(TRAIN_FRACTION * len(data))
    return data[:n], data[n:], stoi, itos


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])  # (B, T)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # (B, T)
    return x.to(device), y.to(device)
