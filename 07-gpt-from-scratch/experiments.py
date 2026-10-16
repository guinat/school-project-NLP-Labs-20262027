import time

import torch

from data import load_splits
from main import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    DROPOUT,
    EVAL_ITERS,
    LEARNING_RATE,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    NUM_THREADS,
    RESULTS_DIR,
    SEED,
)
from model import GPTLanguageModel
from training import plot_losses, train

EVAL_INTERVAL = 250
ARCHITECTURE_ITERS = 2000
DROPOUT_ITERS = 3000
SMALL_TRAIN_SIZE = 50_000

BASE = {
    "block_size": BLOCK_SIZE,
    "n_embd": N_EMBD,
    "n_head": N_HEAD,
    "n_layer": N_LAYER,
    "dropout": DROPOUT,
}
ARCHITECTURE_RUNS = [
    ("block_size=16", {**BASE, "block_size": 16}),
    ("baseline", BASE),
    ("block_size=64", {**BASE, "block_size": 64}),
    ("n_head=1", {**BASE, "n_head": 1}),
]
DROPOUT_RUNS = [
    ("dropout=0.0", {**BASE, "dropout": 0.0}),
    ("dropout=0.2", {**BASE, "dropout": 0.2}),
]
COLUMNS = ["run", "params", "train loss", "val loss", "best val loss", "time (s)"]


def run(config, max_iters, train_data, val_data, vocab_size):
    torch.manual_seed(SEED)
    model = GPTLanguageModel(
        vocab_size,
        config["n_embd"],
        config["n_head"],
        config["n_layer"],
        config["block_size"],
        config["dropout"],
    ).to(DEVICE)
    start = time.time()
    history = train(
        model,
        train_data,
        val_data,
        batch_size=BATCH_SIZE,
        max_iters=max_iters,
        learning_rate=LEARNING_RATE,
        eval_interval=EVAL_INTERVAL,
        eval_iters=EVAL_ITERS,
        device=DEVICE,
    )
    elapsed = time.time() - start
    n_params = sum(p.numel() for p in model.parameters())
    return history, n_params, elapsed


def run_all(runs, max_iters, train_data, val_data, vocab_size):
    rows = []
    histories = {}
    for name, config in runs:
        print(f"\n{name}")
        history, n_params, elapsed = run(
            config, max_iters, train_data, val_data, vocab_size
        )
        _, train_loss, val_loss = history[-1]
        best_step, _, best_val = min(history, key=lambda h: h[2])
        rows.append(
            [
                name,
                f"{n_params:,}",
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{best_val:.4f} (step {best_step})",
                f"{elapsed:.0f}",
            ]
        )
        histories[name] = history
    return rows, histories


def to_markdown(rows):
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    # petites matrices: des threads CPU en plus coûtent plus qu'ils ne font gagner
    torch.set_num_threads(NUM_THREADS)
    RESULTS_DIR.mkdir(exist_ok=True)
    train_data, val_data, stoi, _ = load_splits()

    arch_rows, _ = run_all(
        ARCHITECTURE_RUNS, ARCHITECTURE_ITERS, train_data, val_data, len(stoi)
    )
    # petit train set: le modèle apprend par cœur, donc le dropout a de quoi corriger
    dropout_rows, dropout_histories = run_all(
        DROPOUT_RUNS,
        DROPOUT_ITERS,
        train_data[:SMALL_TRAIN_SIZE],
        val_data,
        len(stoi),
    )

    report = (
        f"## Context length and number of heads, {ARCHITECTURE_ITERS} steps\n\n"
        + to_markdown(arch_rows)
        + f"\n\n## Dropout, first {SMALL_TRAIN_SIZE:,} training characters, "
        + f"{DROPOUT_ITERS} steps\n\n"
        + to_markdown(dropout_rows)
        + "\n"
    )
    print()
    print(report)
    (RESULTS_DIR / "experiments.md").write_text(report, encoding="utf-8")
    plot_losses(dropout_histories, RESULTS_DIR / "dropout.png")


if __name__ == "__main__":
    main()
