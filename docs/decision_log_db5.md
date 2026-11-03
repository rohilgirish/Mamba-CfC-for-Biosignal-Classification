# Ninapro DB5 Decision Log & Methodological Audit

This document records all methodological decisions and protocol configurations specific to Ninapro DB5, ensuring scientific rigor and transparent comparison against the DB1 reference implementation.

## Decisions

| Category | Decision | Rationale & Context |
|---|---|---|
| **Label Variable** | `restimulus` & `rerepetition` | Replicates DB1 protocol. `restimulus` aligns movement onset and offset with actual physical muscle activation (rather than screen visual prompt `stimulus`), producing a realistic ~62.5% rest distribution. |
| **Exercise Metadata Parsing** | Filename Parsing (`S{sub}_E{ex}_A1.mat`) | DB5 `.mat` files contain a circular permutation defect in the internal variable `mat['exercise']` (E1 contains 3, E2 contains 1, E3 contains 2). Parsing strictly from the filename correctly recovers the true Ninapro exercise structure. |
| **Label Space Mapping** | Unified 53-class mapping ($0 \dots 52$) | - Rest: $0 \to 0$<br>- Exercise 1 (12 gestures): $1 \dots 12$<br>- Exercise 2 (17 gestures): $13 \dots 29$ (offset $+12$)<br>- Exercise 3 (23 gestures): $30 \dots 52$ (offset $+29$)<br>Forms a contiguous 0..52 space with zero class collision. |
| **Filtering Protocol** | None (Z-score normalization only) | Follows the baseline protocol "no software filtering". **Crucial distinction**: DB1 signals were on-chip analog envelopes (Otto Bock, unipolar, ~100 Hz), whereas DB5 signals are raw bipolar sEMG (Thalmic Myo, -128 to +127, 200 Hz). Consequently, cross-dataset accuracy numbers are not directly comparable without software envelope extraction. |
| **Sampling Rate & Windowing** | 200 Hz, Window = 60 samples, Stride = 20 samples | 200 Hz is explicitly verified from `mat['frequency']`. 300 ms window $= 0.300 \times 200 = 60$ samples. 100 ms stride $= 0.100 \times 200 = 20$ samples. |
| **Disjoint Subject Split** | 6 Train / 2 Val / 2 Test (Seed 42) | Deviation from exact 70/15/15 due to 10 total subjects. An exact 7/1.5/1.5 split is impossible. Option B (6/2/2) was chosen over 7/1/2 to provide 2 subjects in validation, eliminating single-subject variance during early stopping. Seed 42 permutation yields:<br>- **Train**: `[9, 2, 6, 1, 8, 3]` (60% subjects, 186,942 windows)<br>- **Val**: `[10, 5]` (20% subjects, 65,414 windows)<br>- **Test**: `[4, 7]` (20% subjects, 65,237 windows) |
| **Channel Inductive Bias** | 16 Flat Channels | Channels 1..8 (proximal armband) and 9..16 (distal armband) are treated as flat channels into `nn.Linear(16, d_model)` without spatial inductive priors, exactly matching DB1's 10-channel projection. |
| **Boundary Handling** | Per-file windowing + Last-sample rule | Windowing is strictly bounded within each recording file (no windows cross exercise or subject boundaries). Windows crossing internal repetition boundaries are preserved with labels determined by the last sample in the window. |
| **Width Match $W$** | $W = 88$ | Recomputed parameter search for $d_{in}=16, C=53$ gives $122,901$ parameters for `mamba-width` ($W=88$), matching `mamba-cfc` ($123,637$) within $-0.60\%$ ($-736$ parameters). |

