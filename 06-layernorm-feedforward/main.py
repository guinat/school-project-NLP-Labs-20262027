import torch
import torch.nn as nn

N_TOKENS = 32
N_EMBD = 100
B, T, C = 4, 8, 32
HIDDEN_MULTS = (2, 4, 8)
CHANGED_TOKEN = 3


class LayerNorm1d:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(dim=-1, keepdim=True)  # (..., 1)
        xvar = x.var(dim=-1, keepdim=True, unbiased=False)  # (..., 1)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # (..., dim)
        return self.gamma * xhat + self.beta  # (..., dim)


class FeedForward(nn.Module):
    def __init__(self, n_embd, mult=4, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            activation(),
            nn.Linear(mult * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)  # (..., n_embd)


def hidden_activations(ffwd, x):
    return ffwd.net[:2](x)  # (..., mult * n_embd)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def layernorm_axis():
    torch.manual_seed(1337)
    module = LayerNorm1d(N_EMBD)
    x = torch.randn(N_TOKENS, N_EMBD)  # (32, 100)
    y = module(x)  # (32, 100)

    print(f"y[:, 0]: mean {y[:, 0].mean():.4f}, std {y[:, 0].std():.4f}")
    print(f"y[0, :]: mean {y[0, :].mean():.4f}, std {y[0, :].std():.4f}")
    max_token_mean = y.mean(dim=-1).abs().max()
    max_feature_mean = y.mean(dim=0).abs().max()
    print(
        f"max |mean| per token: {max_token_mean:.2e}, "
        f"per feature: {max_feature_mean:.2e}"
    )

    ref = nn.LayerNorm(N_EMBD)(x)  # (32, 100)
    diff = (y - ref).abs().max()
    match = torch.allclose(y, ref, atol=1e-6)
    print(f"allclose with nn.LayerNorm: {match} (max diff {diff:.2e})")

    x_3d = torch.randn(B, T, C)  # (B, T, C)
    match_3d = torch.allclose(LayerNorm1d(C)(x_3d), nn.LayerNorm(C)(x_3d), atol=1e-6)
    print(f"allclose on {tuple(x_3d.shape)}: {match_3d}")

    constant = torch.full((1, N_EMBD), 3.0)  # (1, 100)
    for eps in (1e-5, 0.0):
        out = LayerNorm1d(N_EMBD, eps=eps)(constant)  # (1, 100)
        print(f"constant token, eps={eps}: max |y| = {out.abs().max():.4f}")


def feedforward():
    torch.manual_seed(1337)
    ffwd = FeedForward(N_EMBD)
    x = torch.randn(N_TOKENS, N_EMBD)  # (32, 100)
    hidden = hidden_activations(ffwd, x)  # (32, 400)
    delta = ffwd(x)  # (32, 100)

    print(f"hidden: {tuple(hidden.shape)}")
    print(f"delta: {tuple(delta.shape)}")
    print(f"positive hidden activations: {(hidden > 0).float().mean():.3f}")


def prenorm_block():
    torch.manual_seed(1337)
    layer_norm = nn.LayerNorm(C)
    ffwd = FeedForward(C)
    x = torch.randn(B, T, C)  # (B, T, C)
    x_norm = layer_norm(x)  # (B, T, C)
    delta = ffwd(x_norm)  # (B, T, C)
    x_out = x + delta  # (B, T, C)

    print(f"x: {tuple(x.shape)}")
    print(f"delta: {tuple(delta.shape)}")
    print(f"x_out: {tuple(x_out.shape)}")

    print("\nactivation  positive  non-zero")
    for activation in (nn.ReLU, nn.GELU):
        torch.manual_seed(1337)  # mêmes poids pour les deux activations
        model = FeedForward(C, activation=activation)
        hidden = hidden_activations(model, x_norm)  # (B, T, 4C)
        positive = (hidden > 0).float().mean()
        nonzero = (hidden != 0).float().mean()
        print(f"{activation.__name__:<10}  {positive:8.3f}  {nonzero:8.3f}")

    print("\nhidden  params  2*C*h + h + C  output")
    for mult in HIDDEN_MULTS:
        h = mult * C
        model = FeedForward(C, mult=mult)
        out = model(x_norm)  # (B, T, C)
        formula = 2 * C * h + h + C
        print(f"{h:>6}  {count_params(model):>6}  {formula:>13}  {tuple(out.shape)}")
    print(f"layer norm params: {count_params(layer_norm)}")

    print("\n              shape       cos(x, out)  |out - x| / |x|")
    for name, out in (("with residual", x_out), ("no residual", delta)):
        cos = torch.cosine_similarity(x, out, dim=-1).mean()
        rel = (out - x).norm() / x.norm()
        print(f"{name:<13} {tuple(out.shape)}  {cos:11.3f}  {rel:15.3f}")

    x_changed = x.clone()  # (B, T, C)
    x_changed[:, CHANGED_TOKEN] = torch.randn(B, C)  # (B, C)
    change = (ffwd(layer_norm(x_changed)) - delta).abs().amax(dim=(0, 2))  # (T,)
    print(f"\nmax change of delta per position after editing token {CHANGED_TOKEN}:")
    print(" ".join(f"{v:.4f}" for v in change.tolist()))


def main():
    layernorm_axis()
    print()
    feedforward()
    print()
    prenorm_block()


if __name__ == "__main__":
    main()
