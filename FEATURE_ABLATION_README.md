# Feature Ablation Study — ARCA-Net v2

> **Status: complete.** All 29 configs finished on the deduplicated RAVDESS corpus
> (1,431 files). Results below are real.
>
> **Important data-integrity note**: an earlier run of this study (and an earlier
> RAVDESS/IEMOCAP/EMODB comparison) used a `speech/RAVDESS/` folder that
> accidentally contained the corpus **twice** (once flat under `Actor_01..24`,
> once again nested under `audio_speech_actors_01-24/Actor_01..24`) — 2,865 files
> instead of the correct ~1,440. With near-duplicate audio scattered across an
> 80/20 split, a large fraction of the "held-out" test set had its literal twin
> in the training set, which is why that run reported suspiciously high numbers
> (95%+ fine accuracy). That duplicate folder has been deleted and every number
> below comes from the clean, deduplicated corpus. Baseline here (70.4% fine /
> 79.1% coarse) is lower than the paper's reported 80.56% because this ablation
> uses a **reduced training budget** (single seed, 35 epochs) instead of the
> paper's full 90-epoch × 5-seed Dirichlet ensemble — by design, to make 29 runs
> tractable. Treat all numbers here as relative comparisons between configs, not
> as the final production accuracy.

## What This Is

An ablation study testing 7 new advanced acoustic/dynamical feature families
against the existing ARCA-Net v2 baseline (320 handcrafted features), on RAVDESS.
Each family is tested **alone**, then in **every pairwise combination** with every
other family (28 configs total, plus the baseline = 29 configs), to see which
features actually help fine (8-class) and coarse (4-class) emotion classification,
and which hurt.

- **Dataset**: RAVDESS only (2,865 utterances)
- **Split**: identical 80/20 train/test split reused across all 29 configs (so
  accuracy differences reflect feature content, not split variance)
- **Training budget per config**: single seed (42), 35 epochs, batch size 64,
  AdamW + OneCycleLR — a reduced budget relative to the production model's
  90-epoch × 5-seed ensemble, chosen to make 29 runs tractable. The winning
  configuration(s) should be re-trained with the full production protocol before
  being treated as final numbers.
- **Model**: `ARCANetV2Ablation` — architecturally identical to the production
  `ARCANetV2` when no extra features are added; gains one additional encoder
  branch ("extra_enc") that consumes whichever feature family/families are being
  tested, feeding into the same transformer-fusion + capsule-routing head as the
  existing spectral/relational/prosodic branches.

## Why a New Branch, Not Just Extra Columns

The production model's `forward()` slices its input by three **fixed** offsets:
`spec_dim=256, rel_dim=38, pros_dim=26` (summing to exactly 320, matching
`extract_enhanced_features()`'s output one-to-one). Appending new columns past
320 would be silently ignored by the existing architecture. So the ablation
model adds a 4th branch sized to whatever the active feature family/families
require, only when `extra_dim > 0`.

## The Baseline (320 features)

The existing handcrafted feature set, unchanged:

| Group | Dims | Contents |
|---|---|---|
| MFCC + deltas | 120 | 20 MFCCs × (mean, std) × (raw, Δ, ΔΔ) |
| Mel spectrogram | 128 | 64 mel bins × (mean, std) in dB |
| Spectral misc | 8 | centroid, bandwidth, rolloff, spectral entropy (mean/std each) |
| Chroma | 24 | 12 chroma bins × (mean, std) |
| Spectral contrast | 14 | 7 bands × (mean, std) |
| Voice quality | 10 | pitch stats, HNR, TKEO, jitter, shimmer |
| Energy + ZCR | 4 | RMS mean/std, zero-crossing-rate mean/std |
| Visibility graph (existing) | 12 | degree/edge/hub stats from a natural-visibility-graph on the RMS envelope |

This 320-dim baseline feeds the model's existing spec(256)/rel(38)/pros(26)
branches exactly as in production.

## The 7 New Feature Families

All 7 are computed from the same downsampled, z-scored RMS energy envelope of
each clip (except Information Geometry, which uses per-frame mel-spectrogram
distributions) — the same base signal the existing visibility-graph block
already uses, so the new families are directly comparable extensions of that
same "envelope dynamics" idea rather than a totally different data source.
Every extractor emits **fixed-length summary statistics** (never raw
positional values), so output size never depends on clip duration.

### 1. Koopman / Dynamic Mode Decomposition (`koopman`, 10 dims)
Treats the envelope as a nonlinear dynamical system and linearizes it via DMD:
builds a delay-embedding (Hankel) matrix, computes the best-fit linear operator
between consecutive snapshots via truncated SVD, and eigendecomposes it.
- Dominant eigenvalue magnitude (mean, std, max) — dominant oscillatory/decay modes
- Eigenvalue phase (mean, std) — dominant frequency content of the dynamics
- Stability margin `|λ|-1` (mean, std) — how quickly modes grow/decay
- Mode energy entropy, mean, std — how energy is spread across dynamic modes

### 2. Graph Signal Processing (`gsp`, 8 dims)
Builds a banded similarity graph over the envelope (edges between temporally
nearby points, weighted by Gaussian similarity), then analyzes its Laplacian.
- Laplacian eigenvalue mean, std, max
- Algebraic connectivity (smallest non-zero eigenvalue)
- Eigenvalue spread (max − min)
- Spectral entropy of the eigenvalue distribution
- Signal smoothness (`xᵀLx / xᵀx`) — how much the envelope varies across graph edges
- Total variation — normalized sum of squared edge differences

### 3. Topological Data Analysis (`tda`, 8 dims)
Delay-embeds the envelope into a 3D point cloud (Takens embedding) and
extracts simplified persistent-homology-style descriptors (self-contained —
no `ripser`/`gudhi` dependency):
- Betti-0 (connected components) and Betti-1 (independent loops, via graph
  cyclomatic number) at two distance thresholds, each normalized by point count
- Persistence entropy of the H0 merge-distance (MST) distribution
- H0 lifetime mean, std, max

### 4. Hamiltonian / Lagrangian Dynamics (`hamiltonian`, 8 dims)
Models the envelope as a position `x(t)` with velocity `v(t) = dx/dt`, and
computes a harmonic-oscillator-style energy `H = 0.5v² + 0.5x²`.
- Energy mean, std, max
- Energy transfer rate: `|dH/dt|` mean, std
- Dissipation coefficient — slope of `log(H)` over time (decay/growth rate)
- Phase–energy coupling — correlation between phase angle `atan2(v,x)` and `H`
- Phase angle spread (std)

### 5. Recurrence Quantification Analysis (`rqa`, 5 dims)
Delay-embeds the envelope, builds a recurrence matrix (points within a distance
threshold), and quantifies its structure — the classic RQA metric set:
- **RR** (Recurrence Rate) — overall density of recurrence points
- **DET** (Determinism) — fraction of recurrence points forming diagonal lines
  (≥2 long), i.e. deterministic/predictable structure
- **LAM** (Laminarity) — fraction forming vertical lines, i.e. "stuck" states
- **TT** (Trapping Time) — average vertical line length
- Entropy of diagonal line lengths — complexity of the deterministic structure

### 6. Visibility Graph v2 (`visibility_graph_v2`, 6 dims)
A **distinct, more graph-theoretic** extension of the existing 12-dim
visibility-graph block — built with `networkx` for exact clustering/path-length
computation rather than just degree/edge-length statistics:
- Degree mean, std, entropy
- Average clustering coefficient
- Average shortest path length (largest connected component)
- Network entropy (spectral entropy of the normalized Laplacian)

### 7. Information Geometry (`info_geometry`, 9 dims)
Treats each mel-spectrogram frame as a probability distribution over mel bins
and analyzes how that distribution evolves across the utterance:
- Fisher–Rao distance between consecutive frames (mean, std) — a proper
  Riemannian-geometry distance on the probability simplex (via the
  Bhattacharyya-coefficient closed form)
- Distance spread (max − min) across the utterance
- Fisher–Rao distance between the utterance's early quarter and late quarter
  ("distance between emotional states" proxy within a single clip)
- Discrete curvature of the distance trajectory (mean, std)
- KL divergence between consecutive frames (mean, std)
- Rényi divergence (α=0.5) between consecutive frames (mean)

**Total if all 7 combined**: 10+8+8+8+5+6+9 = **54 extra dimensions** on top of
the 320-dim baseline.

## Ablation Design

- **Configs tested**: 1 baseline + 7 singles + C(7,2)=21 pairs = **29 total**
- **Fixed train/test split** across all configs (stratified 80/20 on fine
  label, seed 42) — isolates the effect of feature content from split variance
- **Feature extraction cached once** (`ablation_features_cache.pth`) — all 29
  configs are built by concatenating cached per-family vectors, so the
  (cheap, ~a few minutes for all 2,865 files × 7 families) extraction pass
  only happens once, not 29 times
- **Per-config training**: single seed, 35 epochs, `HierarchicalMarginLoss`
  (weighted fine + coarse cross-entropy + CapsNet-style margin term — see
  note below), best-epoch fine/coarse accuracy recorded

> **Note on `HierarchicalMarginLoss`**: this class was referenced by
> `arca_train.py`'s import but was missing from `utils.py` entirely — the
> production training script was broken with an `ImportError` before this
> study started. It has been implemented (weighted fine CE + weighted coarse
> CE × 0.3 + CapsNet-style margin loss on `sigmoid(fine_logits)` × 0.4,
> matching the parameters `arca_train.py` was already passing) and verified
> with a forward/backward smoke test. This fixes the main training path, not
> just the ablation harness.

---

## Results — Single Features (vs. Baseline)

**Baseline: Fine 70.38% | Coarse 79.09%**

| Rank | Feature Family | Extra Dims | Fine Acc | Δ Fine | Coarse Acc | Δ Coarse |
|---|---|---|---|---|---|---|
| — | **Baseline (no additions)** | 0 | 70.38% | — | 79.09% | — |
| 1 | Hamiltonian / Lagrangian | 8 | **71.78%** | +1.39 | 78.75% | −0.35 |
| 2 | Information Geometry | 9 | 71.43% | +1.05 | 78.75% | −0.35 |
| 3 | Recurrence Quantification (RQA) | 5 | 70.73% | +0.35 | 77.00% | −2.09 |
| 4 | Topological Data Analysis | 8 | 70.38% | +0.00 | 78.75% | −0.35 |
| 5 | Koopman / DMD | 10 | 70.03% | −0.35 | 79.09% | +0.00 |
| 5 | Visibility Graph v2 | 6 | 70.03% | −0.35 | 77.70% | −1.39 |
| 7 | Graph Signal Processing | 8 | 69.34% | −1.05 | 77.35% | −1.74 |

**Alone, no single feature helps both metrics.** Hamiltonian and Information
Geometry give the clearest fine-grained lift (+1.4 / +1.1 points) but both cost
a small amount of coarse accuracy. Koopman is the only family that doesn't hurt
coarse accuracy at all (flat), but it also doesn't help fine accuracy. GSP is
the weakest family alone, hurting both metrics.

## Results — All Pairwise Combinations

Ranked by fine accuracy. "Δ base" = vs. the 70.38%/79.09% baseline. "Δ best" =
vs. whichever of the pair's two single-family scores was higher alone — a
positive Δ best means the combination is **additive** (genuinely complementary
information); a Δ best near zero or negative means the pair is **redundant**
or **interfering**.

| Rank | Combination | Extra Dims | Fine Acc | Δ base | Δ best single | Coarse Acc | Δ base | Δ best single |
|---|---|---|---|---|---|---|---|---|
| 1 | **Hamiltonian + RQA** | 13 | **75.26%** | **+4.88** | **+3.48** | 81.88% | +2.79 | +3.14 |
| 2 | GSP + Visibility Graph v2 | 14 | 73.87% | +3.48 | +3.83 | 80.14% | +1.05 | +2.44 |
| 3 | Hamiltonian + Visibility Graph v2 | 14 | 73.52% | +3.14 | +1.74 | 80.49% | +1.39 | +1.74 |
| 3 | TDA + RQA | 13 | 73.52% | +3.14 | +2.79 | 79.79% | +0.70 | +1.05 |
| 5 | GSP + TDA | 16 | 73.17% | +2.79 | +2.79 | 79.09% | +0.00 | +0.35 |
| 6 | GSP + RQA | 13 | 72.47% | +2.09 | +1.74 | **82.23%** | **+3.14** | **+4.88** |
| 6 | RQA + Information Geometry | 14 | 72.47% | +2.09 | +1.05 | 79.09% | +0.00 | +0.35 |
| 8 | Visibility Graph v2 + Info Geometry | 15 | 71.43% | +1.05 | +0.00 | 78.05% | −1.05 | −0.70 |
| 8 | TDA + Hamiltonian | 16 | 71.43% | +1.05 | −0.35 | 75.96% | −3.14 | −2.79 |
| 8 | Hamiltonian + Info Geometry | 17 | 71.43% | +1.05 | −0.35 | 77.00% | −2.09 | −1.74 |
| 11 | Koopman + Info Geometry | 19 | 70.73% | +0.35 | −0.70 | 79.44% | +0.35 | +0.35 |
| 12 | Koopman + GSP | 18 | 70.03% | −0.35 | +0.00 | 76.66% | −2.44 | −2.44 |
| 12 | Koopman + Hamiltonian | 18 | 70.03% | −0.35 | −1.74 | 75.61% | −3.48 | −3.48 |
| 12 | RQA + Visibility Graph v2 | 11 | 70.03% | −0.35 | −0.70 | 79.44% | +0.35 | +1.74 |
| 12 | TDA + Visibility Graph v2 | 14 | 70.03% | −0.35 | −0.35 | 80.14% | +1.05 | +1.39 |
| 12 | Koopman + Visibility Graph v2 | 16 | 70.03% | −0.35 | +0.00 | 79.79% | +0.70 | +0.70 |
| 12 | GSP + Hamiltonian | 16 | 70.03% | −0.35 | −1.74 | 77.35% | −1.74 | −1.39 |
| 18 | Koopman + RQA | 15 | 69.69% | −0.70 | −1.05 | 79.09% | +0.00 | +0.00 |
| 18 | Koopman + TDA | 18 | 69.69% | −0.70 | −0.70 | 78.75% | −0.35 | −0.35 |
| 18 | GSP + Info Geometry | 17 | 69.69% | −0.70 | −1.74 | 77.70% | −1.39 | −1.05 |
| 18 | TDA + Info Geometry | 17 | 69.69% | −0.70 | −1.74 | 78.40% | −0.70 | −0.35 |

## Analysis

**Hamiltonian + RQA is the clear winner** — the only config that improves both
fine accuracy (+4.88 pts) and coarse accuracy (+2.79 pts) substantially over
baseline, and it beats *both* of its constituent single features by a wide
margin (+3.48 over Hamiltonian alone, +4.53 over RQA alone). This is a clean
example of **additive, complementary signal**: Hamiltonian energy dynamics
capture how the envelope's energy grows/decays and how phase and energy couple,
while RQA captures a different property — how deterministic/recurrent the
envelope's trajectory is (diagonal/vertical line structure). Neither subsumes
the other, and combined they give the model two independent views of the same
underlying nonlinear dynamics.

**GSP + RQA is the best config specifically for coarse accuracy** (82.23%,
+3.14 over baseline, +4.88 over the better single feature) despite GSP being
the *worst*-performing single feature alone. This is a striking redundancy-
breaking result: GSP's graph-Laplacian view of the envelope apparently encodes
information that's nearly useless on its own but strongly complementary to
RQA's recurrence structure for the *coarse* 4-group split specifically.

**Koopman is the standout disappointment.** Alone it's roughly neutral
(slightly hurts fine, flat on coarse), but in combination it actively
*interferes* more often than not — most notably **Koopman + Hamiltonian**,
where Hamiltonian alone was the single best feature (+1.39 fine) but adding
Koopman drags coarse accuracy down by −3.48 points versus baseline and −3.48
versus Hamiltonian alone. The same pattern (small fine gain wiped out by a
coarse-accuracy hit) shows up in Koopman+GSP. A plausible explanation: the
Koopman/DMD eigenvalue-based features are themselves derived from a delay-
embedding SVD of the same RMS envelope that several other families also use
(Hamiltonian, RQA, Visibility Graph v2) — when two families extract highly
correlated information but with different noise characteristics, the extra
branch's parameters can fit noise rather than signal on a training set this
size (~1,145 samples).

**TDA + Hamiltonian** shows a similar interference pattern to Koopman+Hamiltonian:
Hamiltonian's fine-accuracy gain survives (71.43%) but its coarse accuracy is
badly hurt (−3.14 vs baseline, −2.79 vs Hamiltonian alone). Hamiltonian appears
to be a fragile feature that only combines well with RQA or Visibility-Graph-v2,
not with Koopman, GSP, or TDA.

**Practical takeaway for the final feature set:**
- **Best overall**: `baseline (320) + Hamiltonian (8) + RQA (5)` = 333 dims,
  the config to carry into a full 90-epoch × 5-seed production retraining.
- **Worth a second full run for comparison**: `baseline + GSP + RQA` (326 dims)
  if coarse-grained accuracy matters more for the paper's framing than fine-grained.
- **Do not include Koopman** in the final feature set based on this evidence —
  it never produces a clear win and actively hurts several otherwise-strong
  combinations.
- These are single-seed, 35-epoch numbers on ~1,145 training samples; before
  reporting a final number in the paper, re-run the winning config(s) with the
  full production protocol (90 epochs, 5-seed Dirichlet ensemble) to confirm
  the gain holds up and isn't an artifact of the reduced budget's higher
  variance.

---

*Generated by `ablation_study.py`. Raw per-config results:
`results/ablation_study/ablation_results.csv`. Feature extractors:
`advanced_features.py`. Ablation model: `arca_net_ablation.py`.*
