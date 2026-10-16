import time
from pathlib import Path

import torch

from data import decode, load_splits
from model import GPTLanguageModel
from training import plot_losses, train

BATCH_SIZE = 16
BLOCK_SIZE = 32
MAX_ITERS = 5000
EVAL_INTERVAL = 100
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0
SEED = 1337
NUM_THREADS = 1

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    # petites matrices: des threads CPU en plus coûtent plus qu'ils ne font gagner
    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"device: {DEVICE}")

    train_data, val_data, stoi, itos = load_splits()
    print(f"vocab size: {len(stoi)}")
    print(f"train tokens: {len(train_data)}, val tokens: {len(val_data)}")

    model = GPTLanguageModel(
        len(stoi), N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params} ({n_params / 1e6:.3f}M)")

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print("\nsample before training:")
    print(decode(model.generate(context, 300)[0].tolist(), itos))

    print()
    start = time.time()
    history = train(
        model,
        train_data,
        val_data,
        batch_size=BATCH_SIZE,
        max_iters=MAX_ITERS,
        learning_rate=LEARNING_RATE,
        eval_interval=EVAL_INTERVAL,
        eval_iters=EVAL_ITERS,
        device=DEVICE,
    )
    print(f"training time: {time.time() - start:.0f}s")
    plot_losses({"baseline": history}, RESULTS_DIR / "loss.png")

    model.eval()
    sample = decode(model.generate(context, 2000)[0].tolist(), itos)
    (RESULTS_DIR / "sample.txt").write_text(sample, encoding="utf-8")
    print("\nsample after training:")
    print(sample)


if __name__ == "__main__":
    main()
