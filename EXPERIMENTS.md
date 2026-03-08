# Experiments Log — Sentence Decoding Architecture

Tracks every architectural change, why it was made, and the results vs baseline.

---

## Baseline

**Date:** 2026-03-04
**Config:** Default `sentence_decoding/grids/defaults.py`, no architecture changes
**Dataset:** Gwilliams2022, sub-01 only (~4.3 GB, 362 words)
**Epochs:** Early stopped at epoch 10 / 50 (patience=10)

**Architecture:**
```
Input [batch, 208, 150]
  → initial_linear: Conv1d(208 → 512, kernel=1)
  → subject_layers: Linear(512 → 512) per-subject
  → encoder: 5× Conv1d with dilation + GLU + skip connections
  → BahdanauAttention: collapses time → [batch, hidden]
  → TransformerEncoder: 16-layer, 16-head cross-batch transformer
  → output: [batch, 1024]
Total: 610M params
```

**Test results:**
| Metric | Score |
|---|---|
| test_retrieval_acc1 (top-1 / 250) | 2.3% |
| test_retrieval_acc10 (top-10 / 250) | 23.3% |
| test_retrieval_rank (median) | 18.0 |
| test_contrastive_top1_acc | 2.3% |
| test_contrastive_top5_acc | 14.0% |
| test_contrastive_cosine_sim | 0.522 |
| test_sentence_bert | 0.300 |
| test_transformer_loss | 5.14 |
| test_cnn_loss | 7.79 |

Baseline results saved in full at: `baseline_results.txt`

---

## Experiment 1: Spatial Filter (Beamformer) Layer

**Date:** 2026-03-04
**Status:** Running (task b7avyf1t8)

### Motivation

The baseline CNN uses `Conv1d` which slides filters along the **time axis only**.
It processes each MEG channel independently and never explicitly learns *which
combination of the 208 sensors* is most informative for a given brain state.

MEG has strong spatial structure — sensors are arranged in a helmet over the head,
and brain regions activate in clusters across nearby sensors. A spatial filter
("beamformer") learns to weight and combine sensors before any temporal processing,
analogous to how ICA or beamforming is done in traditional MEG preprocessing.

This is the key spatial step from **EEGNet** and **EEG Conformer** (Song et al., IEEE).

### What was changed

**File 1: `neuraltrain/neuraltrain/models/simpleconv.py`**

Added new class `SpatialFilter`:
```python
class SpatialFilter(nn.Module):
    def __init__(self, n_channels: int, n_filters: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters, kernel_size=(n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)   # [B, 1, C, T]
        x = self.conv(x)     # [B, n_filters, 1, T]
        x = self.bn(x)
        return x.squeeze(2)  # [B, n_filters, T]
```

Added config option to `SimpleConvConfig`:
```python
spatial_filters: int = 0  # 0 = disabled; e.g. 32 to learn 32 channel combinations
```

Wired into `SimpleConv.__init__` (inserted after merger, before initial_linear):
```python
self.spatial_filter = None
if config.spatial_filters > 0:
    self.spatial_filter = SpatialFilter(in_channels, config.spatial_filters)
    in_channels = config.spatial_filters
```

Wired into `SimpleConv.forward`:
```python
if self.spatial_filter is not None:
    x = self.spatial_filter(x)
```

**File 2: `sentence_decoding/grids/defaults.py`**

Added to `brain_model_config`:
```python
"spatial_filters": 32,
```

### New data flow
```
Input [batch, 208, 150]
  → SpatialFilter: Conv2d(208→32, kernel=(208,1)) + BN   ← NEW (~6,700 params)
  → initial_linear: Conv1d(32 → 512, kernel=1)           ← input dim changed: 208→32
  → subject_layers: Linear(512 → 512)
  → encoder: 5× Conv1d with dilation + GLU + skip
  → BahdanauAttention: collapses time → [batch, hidden]
  → TransformerEncoder: 16-layer, 16-head
  → output: [batch, 1024]
```

### Parameter count change
- SpatialFilter adds: 32 × 208 = 6,656 weights + 64 BN params = **~6,720 params**
- initial_linear changes: 208×512 → 32×512 = saves **~90,000 params**
- Net change: **slightly fewer params overall**

### Results

| Metric | Baseline | Exp 1 (+SpatialFilter) | Δ |
|---|---|---|---|
| test_retrieval_acc1 | 2.3% | 2.3% | 0 |
| test_retrieval_acc10 | 23.3% | 23.3% | 0 |
| test_retrieval_rank | 18.0 | 18.6 | -0.6 |
| test_contrastive_top1_acc | 2.3% | 2.3% | 0 |
| test_contrastive_top5_acc | 14.0% | 14.0% | 0 |
| test_contrastive_cosine_sim | 0.522 | 0.518 | -0.004 |
| test_sentence_bert | 0.300 | 0.300 | 0 |
| test_transformer_loss | 5.14 | 5.16 | +0.02 |
| test_cnn_loss | 7.79 | 7.89 | +0.10 |

### Conclusion
No meaningful change. As predicted, with only 1 subject (362 words, 30 total gradient
updates), the spatial filter has no signal to learn what channel combinations are
informative. The layer is correctly implemented and wired in — this result does **not**
rule out its effectiveness. Re-evaluate with 5+ subjects.

---

## Ideas for Future Experiments

### Exp 2: Within-sample temporal transformer
The current `TransformerEncoder` operates **across the batch** (128 samples attend to
each other). Adding a transformer that operates **within each sample across time steps**
would capture long-range temporal dependencies inside a single brain recording.

### Exp 3: Frequency-domain features
MEG frequency bands carry distinct information (theta=language, alpha=attention,
beta=syntax, gamma=local computation). Could prepend a learnable filterbank (e.g.
learned sinc filters like SincNet) or feed FFT features alongside raw signal.

### Exp 4: More subjects
Most impactful non-architecture change. Going from 1 → 27 subjects is the largest
possible improvement. Use `download_gwilliams_subject.py` and loop over sub-01..sub-27.

### Exp 5: Larger text encoder
Config already has `facebook/opt-2.7b` commented out in defaults.py as an alternative
to T5-large. A larger LM may produce richer target embeddings.
