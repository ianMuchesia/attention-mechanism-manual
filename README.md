attention-mechanism-manual
==========================

Repository description: A compact, from-scratch walkthrough of scaled dot-product attention with visualization in a notebook.

What this repo contains
- `ScaledDotProductAttention` in src/attention.py with Q, K, V linear projections.
- Scaled scores computed as QK^T / sqrt(d_k).
- Softmax normalization to produce attention weights.
- Weighted sum of values to produce outputs.
- Optional masking for padded tokens via `apply_mask`.
- A notebook that prints tensor shapes and verifies score behavior.
- Seaborn heatmap visualization of attention weights.
- Saved figure at experiments/attention_heatmap.png.

What to build next
- Add a toy sequence-to-sequence copy task.
- Compare attention behavior under different scaling factors.

Key learning: How attention allows dynamic focus on input.
