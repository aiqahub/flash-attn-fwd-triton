"""Multi-head attention in PyTorch.

Run this file to see the tensor shapes at every step of multi-head attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(precision=2, sci_mode=False)


def step(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)  # Q, K, V projections in one matmul
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False, verbose=False):
        B, T, C = x.shape
        if verbose:
            step("STEP 1: Input")
            print(f"x shape (batch B, seq_len T, d_model C) = {tuple(x.shape)}")
            print(f"n_heads = {self.h}, head_dim = C / n_heads = {C} / {self.h} = {self.d}")

        qkv = self.qkv(x)
        if verbose:
            step("STEP 2: One linear layer computes Q, K, V together")
            print(f"qkv = Linear(x): {tuple(x.shape)} -> {tuple(qkv.shape)}  (3 * C = {3 * C})")

        q, k, v = qkv.split(C, dim=-1)
        if verbose:
            step("STEP 3: Split into Q, K, V")
            print(f"q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)}")

        # reshape to (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.h, self.d).transpose(1, 2) for t in (q, k, v))
        if verbose:
            step("STEP 4: Split C into heads -> (B, heads, T, head_dim)")
            print(f"view  -> (B, T, heads, head_dim) = {(B, T, self.h, self.d)}")
            print(f"transpose(1, 2) -> {tuple(q.shape)}")
            print("Each head now runs attention independently on its own slice of size", self.d)

        if verbose:
            # Show the attention weights for one head (computed by hand, for display only)
            scores = q @ k.transpose(-2, -1) / self.d ** 0.5
            if causal:
                mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = scores.softmax(dim=-1)
            step("STEP 5: Attention weights per head = softmax(Q K^T / sqrt(head_dim))")
            print(f"scores/weights shape: {tuple(weights.shape)}  (B, heads, T, T)")
            print(f"Causal mask applied: {causal}")
            print("Weights for batch 0, head 0 (row attends to column):")
            print(weights[0, 0])
            print("Row sums (should be 1):", weights[0, 0].sum(-1))

        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        if verbose:
            step("STEP 6: Weighted sum of V per head (scaled_dot_product_attention)")
            print(f"y shape: {tuple(y.shape)}  (B, heads, T, head_dim)")

        y = y.transpose(1, 2).reshape(B, T, C)  # concatenate heads
        if verbose:
            step("STEP 7: Concatenate heads back together")
            print(f"transpose + reshape -> {tuple(y.shape)}  (B, T, C)")

        out = self.out(y)
        if verbose:
            step("STEP 8: Output projection mixes information across heads")
            print(f"out = Linear(y): {tuple(y.shape)} -> {tuple(out.shape)}")
        return out


def manual_attention(q, k, v, causal=False):
    """Same math as F.scaled_dot_product_attention, written out by hand."""
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    if causal:
        T = q.shape[-2]
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ v


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32)  # batch=2, seq=6, dim=32 (small so the printout is readable)
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    y = mha(x, causal=True, verbose=True)

    step("FINAL")
    print("MultiHeadAttention output shape:", tuple(y.shape), "(same shape as input)")

    step("CHECK: hand-written attention vs PyTorch's built-in kernel")
    q, k, v = (torch.randn(2, 4, 10, 16) for _ in range(3))
    ref = manual_attention(q, k, v, causal=True)
    fast = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print("Max difference:", (ref - fast).abs().max().item())
    print("Manual vs built-in match:", torch.allclose(ref, fast, atol=1e-5))


if __name__ == "__main__":
    main()
