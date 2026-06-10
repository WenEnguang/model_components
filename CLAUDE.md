# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a single-module PyTorch implementation of **Multi-Head Attention with KV Cache** (`MHA_KVCache`). The module is designed for decoder-style autoregressive transformer inference where cached key/value tensors from previous time steps are reused to avoid recomputation.

## Environment

Conda is the package manager (see `.vscode/settings.json`). Required dependency: `pytorch` (`torch`, `torch.nn`).

## Architecture

`MHA_KVCache` (`MHA+KACache.py:4`) implements two modes controlled by `use_cache`:

- **`use_cache=True` (KV Cache mode, default):** Receives new tokens' Q/K/V each forward call, projects them, splits into heads, concatenates with `past_key`/`past_value` along the sequence dimension (`dim=2`), computes attention over the full (cached + new) sequence, and returns the updated full K/V tensors as `past_key_values`. This is the standard autoregressive inference pattern.

- **`use_cache=False` (History accumulation mode):** Appends each input `q` to `self.history_seq` (a Python list stored on the module). Each forward call projects the entire accumulated history through K and V projections to produce the key/value tensors. **Note:** `self.history_seq` is never reset — it persists across calls, leaking state between unrelated sequences. Also `self.history_seq` is a list that accumulates raw pre-projection tensors, which means shape mismatches can occur if input sequence lengths vary.

### Forward pass signature

```python
forward(q, k, v, past_key=None, past_value=None, mask=None)
  # Returns: (output, past_key_values)
  # past_key_values = (full_k, full_v) when use_cache=True
```

- `q, k, v`: input tensors of shape `(batch_size, seq_len, dim)`
- `past_key`, `past_value`: cached K/V from previous step, shape `(batch_size, n_heads, past_len, head_dim)`
- `mask`: attention mask applied via `masked_fill(mask == 0, float('-inf'))`

### Projections

Four `nn.Linear(dim, dim, bias=False)` projections — `q_proj`, `k_proj`, `v_proj`, `o_proj`. `head_dim = dim // n_heads`. Defaults: `dim=512`, `n_heads=8` → `head_dim=64`.

## Known issues

- **`use_cache=False` path:** `self.history_seq` is a persistent list that never gets cleared — causes state leakage and potential shape errors if input shapes vary. It also stores pre-projection tensors, which may be incorrect if K/V projections differ from the cached state.
