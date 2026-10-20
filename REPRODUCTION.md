# Reproduction notes

## Model

- Three-layer GATv2 encoder with 79-dimensional atom features and six-dimensional bond features.
- Mean and max graph pooling produce a 256-dimensional structural representation.
- Signaturizer input: 3,200 dimensions (A1–E5), projected to 256 dimensions.
- Per-dimension gated fusion: `z * h_structure + (1 - z) * h_bio`.
- Pair representation: `[h_A, h_B, h_A*h_B, |h_A-h_B|]` (1,024 dimensions).
- Binary head: DDI existence logit.
- Risk head: three severity logits.

## Main cold-start protocol

- Bemis–Murcko scaffold grouping with `includeChirality=False`.
- Scaffold shuffle seed: 42; approximately 20% of scaffold groups are held out as an independent test group.
- Five development folds; each fold excludes validation-fold drugs from training pairs.
- Adam, learning rate `1e-4`, batch size 256, up to 50 epochs.
- ReduceLROnPlateau (`factor=0.1`, `patience=5`) and early stopping (`patience=10`).
- OHNM negative ratio: 2.0.
- Severity loss weight: 0.5; class weights `[10, 1, 3]`; unknown severity label `-1` is ignored.

The training data, 2026 cohort and trained weights are released separately after manuscript acceptance. The code paths above are retained so that the released artifacts can be placed under the documented directories without changing the model implementation.
