# Attention Mechanism Manual
==========================

A minimal, educational implementation of scaled dot-product attention with visualization and math notes.

## Features

- **ScaledDotProductAttention** (`src/attention.py`): Q/K/V projections, scaled scores, softmax, and weighted value aggregation
- **Masking** (`src/attention.py`): Optional padding mask support via `apply_mask`
- **Notebook Walkthrough** (`notebooks/attention_visualization.ipynb`): Step-by-step inspection of shapes and weights
- **Heatmap Output** (`experiments/attention_heatmap.png`): Saved attention visualization
- **Derivation Notes** (`math-notes/`): Supporting math and references

## Quick Start

```bash
python -m jupyter notebook
```

Open `notebooks/attention_visualization.ipynb` and run the cells in order.

## Project Structure

- `src/` — Core implementation
- `notebooks/` — Analysis and visualization
- `experiments/` — Saved outputs
- `math-notes/` — Derivations and notes
