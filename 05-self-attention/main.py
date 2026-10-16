import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 4
BLOCK_SIZE = 8
TOY_CHANNELS = 2
N_EMBD = 32
HEAD_SIZE = 16


def matmul_aggregation():
    a = torch.tril(torch.ones(3, 3))
    a = a / torch.sum(a, 1, keepdim=True)
    b = torch.randint(0, 10, (3, 2)).float()
    c = a @ b  # (3, 3) @ (3, 2) -> (3, 2)
    return a, b, c


def causal_mean_loops(x):
    B, T, C = x.shape
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xprev = x[b, : t + 1]  # (t + 1, C)
            xbow[b, t] = torch.mean(xprev, dim=0)  # (C,)
    return xbow


def causal_mean_matmul(x):
    T = x.shape[1]
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(1, keepdim=True)  # (T, T)
    return wei @ x  # (T, T) @ (B, T, C) -> (B, T, C)


def causal_mean_softmax(x):
    T = x.shape[1]
    tril = torch.tril(torch.ones(T, T))
    wei = torch.zeros((T, T))
    wei = wei.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)  # (T, T)
    return wei @ x  # (B, T, C)


def self_attention_head(x, head_size):
    _, T, C = x.shape
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)

    k = key(x)  # (B, T, head_size)
    q = query(x)  # (B, T, head_size)
    wei = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
    wei = wei * head_size**-0.5
    tril = torch.tril(torch.ones(T, T))
    wei = wei.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)  # (B, T, T)

    v = value(x)  # (B, T, head_size)
    out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
    return q, k, wei, out


def score_variance(head_size):
    k = torch.randn(BATCH_SIZE, BLOCK_SIZE, head_size)  # (B, T, head_size)
    q = torch.randn(BATCH_SIZE, BLOCK_SIZE, head_size)  # (B, T, head_size)
    raw = q @ k.transpose(-2, -1)  # (B, T, T)
    scaled = raw * head_size**-0.5
    return raw.var().item(), scaled.var().item()


def main():
    torch.manual_seed(42)
    a, b, c = matmul_aggregation()
    print("step 1: aggregation by matrix product")
    print(f"a =\n{a}")
    print(f"b =\n{b}")
    print(f"c = a @ b =\n{c}")

    torch.manual_seed(1337)
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, TOY_CHANNELS)  # (B, T, C)
    xbow = causal_mean_loops(x)
    print("\nstep 2: causal mean with loops")
    for t in (0, 3, 7):
        print(f"xprev shape at t={t}: {tuple(x[0, : t + 1].shape)}")
    print(f"x[0] =\n{x[0]}")
    print(f"xbow[0] =\n{xbow[0]}")

    xbow2 = causal_mean_matmul(x)
    print("\nstep 3: causal mean with a normalized tril matrix")
    print(f"allclose(xbow, xbow2): {torch.allclose(xbow, xbow2, atol=1e-6)}")

    xbow3 = causal_mean_softmax(x)
    print("\nstep 4: causal mean with masked_fill and softmax")
    print(f"allclose(xbow, xbow3): {torch.allclose(xbow, xbow3, atol=1e-6)}")

    torch.manual_seed(1337)
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBD)  # (B, T, C)
    q, k, wei, out = self_attention_head(x, HEAD_SIZE)
    row_sums_ok = torch.allclose(wei.sum(dim=-1), torch.ones(BATCH_SIZE, BLOCK_SIZE))
    no_future = bool((wei.triu(diagonal=1) == 0).all())
    print("\nstep 5: one self-attention head")
    print(f"x shape: {tuple(x.shape)}")
    print(f"q shape: {tuple(q.shape)}")
    print(f"k shape: {tuple(k.shape)}")
    print(f"wei shape: {tuple(wei.shape)}")
    print(f"out shape: {tuple(out.shape)}")
    print(f"every row of wei sums to 1: {row_sums_ok}")
    print(f"weights on future tokens are all 0: {no_future}")
    print("wei[0] =")
    for row in wei[0].tolist():
        print("  ".join(f"{w:.2f}" for w in row))

    print("\nstep 6: softmax stability")
    for head_size in (4, HEAD_SIZE, 64):
        torch.manual_seed(1337)
        raw_var, scaled_var = score_variance(head_size)
        print(
            f"head_size {head_size}: var(q @ k^T) {raw_var:.2f}, "
            f"var(q @ k^T / sqrt(head_size)) {scaled_var:.2f}"
        )
    z = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
    print(f"softmax(z) = {F.softmax(z, dim=-1)}")
    print(f"softmax(8 * z) = {F.softmax(z * 8, dim=-1)}")


if __name__ == "__main__":
    main()
