"""Scaled dot-product attention from scratch in NumPy.

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Run this file to see every intermediate step printed out.
"""

import numpy as np

np.set_printoptions(precision=2, suppress=False, linewidth=120)


def step(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def softmax(x, axis=-1, verbose=False):
    shifted = x - x.max(axis=axis, keepdims=True)  # numerical stability
    e = np.exp(shifted)
    out = e / e.sum(axis=axis, keepdims=True)
    if verbose:
        print(
            "  softmax: subtract row max (so the largest value becomes 0):\n", shifted
        )
        print("  softmax: exponentiate:\n", e)
        print("  softmax: divide by row sum -> probabilities:\n", out)
    return out


def attention(Q, K, V, mask=None, tokens=None, verbose=False):
    d_k = Q.shape[-1]

    # 1. Raw similarity between every query and every key
    raw_scores = Q @ K.T
    if verbose:
        step("STEP 4: Raw scores = Q @ K.T")
        print(f"Shape: {Q.shape} @ {K.T.shape} -> {raw_scores.shape}")
        print(
            "Entry [i, j] = dot(query of token i, key of token j) = how well they match"
        )
        print(raw_scores)

    # 2. Scale so softmax doesn't saturate
    scores = raw_scores / np.sqrt(d_k)
    if verbose:
        step(f"STEP 5: Scale by sqrt(d_k) = sqrt({d_k}) = {np.sqrt(d_k):.2f}")
        print(
            "Keeps scores in a reasonable range so softmax doesn't become one-hot too early"
        )
        print(scores)

    # 3. Optional mask
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
        if verbose:
            step("STEP 5b: Apply mask (blocked positions set to -1e9)")
            print("Mask (True = allowed to look at):\n", mask)
            print("Masked scores (-1e9 shown as a huge negative number):\n", scores)

    # 4. Softmax -> attention weights
    if verbose:
        step("STEP 6: Softmax over each row -> attention weights")
    weights = softmax(scores, axis=-1, verbose=verbose)
    if verbose:
        print("\n  Row sums (each should be 1.0):", weights.sum(axis=-1))
        if tokens is not None:
            for t, row in zip(tokens, weights):
                top = tokens[int(row.argmax())]
                print(f"  '{t}' pays most attention to '{top}' ({row.max():.2f})")

    # 5. Weighted sum of values
    output = weights @ V
    if verbose:
        step("STEP 7: Output = weights @ V")
        print(f"Shape: {weights.shape} @ {V.shape} -> {output.shape}")
        print("Each output row is a weighted average of all rows of V")
        print(output)

        # Show the weighted sum explicitly for the first token
        i = 0
        name = tokens[i] if tokens is not None else f"token {i}"
        print(f"\n  Breaking down the output for '{name}':")
        manual = np.zeros(V.shape[1])
        for j in range(V.shape[0]):
            other = tokens[j] if tokens is not None else f"token {j}"
            print(f"    {weights[i, j]:.2f} * V['{other}'] = {weights[i, j] * V[j]}")
            manual += weights[i, j] * V[j]
        print(f"    sum                = {manual}")
        print(f"    matches output[{i}]? {np.allclose(manual, output[i])}")

    return output, weights


def print_weights(title, tokens, w):
    print(title)
    print("       " + "  ".join(f"{t:>5}" for t in tokens))
    for t, row in zip(tokens, w):
        print(f"{t:>5}: " + "  ".join(f"{p:5.2f}" for p in row))
    print()


def main():
    np.random.seed(0)
    tokens = ["the", "cat", "sat", "down"]
    d_model = 16  # dimension of each token embedding

    step("STEP 1: Token embeddings X")
    X = np.random.randn(len(tokens), d_model)
    print(f"{len(tokens)} tokens, each a {d_model}-dim vector -> X shape {X.shape}")
    for t, row in zip(tokens, X):
        print(f"  {t:>5}: {row}")

    step("STEP 2: Projection matrices W_q, W_k, W_v (learned in a real model)")
    W_q, W_k, W_v = (np.random.randn(d_model, d_model) for _ in range(3))
    print(f"W_q, W_k, W_v each have shape {W_q.shape}")

    step("STEP 3: Project embeddings into Queries, Keys, Values")
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    print(f"Q = X @ W_q -> {Q.shape}  (what each token is looking for)")
    print(Q)
    print(f"\nK = X @ W_k -> {K.shape}  (what each token offers to be matched on)")
    print(K)
    print(f"\nV = X @ W_v -> {V.shape}  (the information each token passes along)")
    print(V)

    # Full (bidirectional) attention, BERT-style
    print("\n\n" + "#" * 70)
    print("# PART A: FULL ATTENTION (every token can see every token)")
    print("#" * 70)
    out, w = attention(Q, K, V, tokens=tokens, verbose=True)
    step("RESULT: Full attention weights")
    print_weights("(row attends to column)", tokens, w)
    print("Output shape:", out.shape, "-> one context-aware vector per token")

    # Causal attention, GPT-style: token i only sees tokens 0..i
    print("\n\n" + "#" * 70)
    print("# PART B: CAUSAL ATTENTION (token i can only see tokens 0..i)")
    print("#" * 70)
    causal = np.tril(np.ones((len(tokens), len(tokens)), dtype=bool))
    out_c, w_c = attention(Q, K, V, mask=causal, tokens=tokens, verbose=True)
    step("RESULT: Causal attention weights")
    print_weights("(upper triangle is masked to 0)", tokens, w_c)


if __name__ == "__main__":
    main()
