import matplotlib.pyplot as plt
import torch

from data import get_batch


@torch.no_grad()
def estimate_loss(model, splits, batch_size, eval_iters, device):
    out = {}
    model.eval()
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, model.block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(
    model,
    train_data,
    val_data,
    batch_size,
    max_iters,
    learning_rate,
    eval_interval,
    eval_iters,
    device,
):
    splits = {"train": train_data, "val": val_data}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, splits, batch_size, eval_iters, device)
            history.append((step, losses["train"], losses["val"]))
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch(train_data, batch_size, model.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history


def plot_losses(histories, path):
    plt.figure(figsize=(7, 4))
    for name, history in histories.items():
        steps, train_losses, val_losses = zip(*history, strict=True)
        line = plt.plot(steps, train_losses, label=f"train, {name}")[0]
        plt.plot(steps, val_losses, "--", color=line.get_color(), label=f"val, {name}")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
