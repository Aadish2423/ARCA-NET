"""
ARCA-Net v2 variant used for the feature-ablation study.

The production ARCANetV2 (arca_net.py) has 3 fixed encoder branches whose
dims sum to exactly 320 (spec=256, rel=38, pros=26), matching
utils.extract_enhanced_features' output one-to-one. Its forward() slices
input columns by those fixed offsets, so simply appending new feature
columns to the input vector would be silently ignored by the model.

ARCANetV2Ablation adds an OPTIONAL 4th branch ("extra_enc") that consumes
whatever additional feature-family columns are being tested. When
extra_dim=0 it behaves identically to the original 3-branch architecture
(same fusion sequence length, same flat_head input dim), so the baseline
(extra_dim=0) ablation config is an apples-to-apples reproduction of the
original model -- only configs that add feature families get the 4th
branch.
"""

import torch
import torch.nn as nn

from arca_net import ResBlock, AttentiveRoutingCapsuleLayer, squash


class ARCANetV2Ablation(nn.Module):

    def __init__(
        self,
        num_coarse=4,
        num_fine=8,
        spec_dim=256,
        rel_dim=38,
        pros_dim=26,
        extra_dim=0,
        hidden_dim=128,
        capsule_dim=24,
        primary_caps_per_branch=4,
        dropout=0.35
    ):
        super().__init__()

        self.spec_dim = spec_dim
        self.rel_dim = rel_dim
        self.pros_dim = pros_dim
        self.extra_dim = extra_dim
        self.num_fine = num_fine
        self.capsule_dim = capsule_dim
        self.primary_caps_per_branch = primary_caps_per_branch

        self.n_branches = 4 if extra_dim > 0 else 3

        # ====================================================
        # BRANCH ENCODERS
        # ====================================================

        self.spec_enc = nn.Sequential(
            nn.Linear(spec_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            ResBlock(hidden_dim, dropout)
        )

        self.rel_enc = nn.Sequential(
            nn.Linear(rel_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            ResBlock(hidden_dim, dropout)
        )

        self.pros_enc = nn.Sequential(
            nn.Linear(pros_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            ResBlock(hidden_dim, dropout)
        )

        if self.n_branches == 4:
            self.extra_enc = nn.Sequential(
                nn.Linear(extra_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                ResBlock(hidden_dim, dropout)
            )

        # ====================================================
        # TRANSFORMER FUSION
        # ====================================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.fusion = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        # ====================================================
        # PRIMARY CAPSULE PROJECTION
        # ====================================================

        self.primary_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(
                hidden_dim,
                primary_caps_per_branch * capsule_dim
            )
        )

        # ====================================================
        # CAPSULE CONTEXT ATTENTION
        # ====================================================

        self.capsule_context = nn.MultiheadAttention(
            embed_dim=capsule_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.context_norm = nn.LayerNorm(capsule_dim)

        # ====================================================
        # EMOTION CAPSULES
        # ====================================================

        self.emotion_caps = AttentiveRoutingCapsuleLayer(
            in_caps=self.n_branches * primary_caps_per_branch,
            out_caps=num_fine,
            in_dim=capsule_dim,
            out_dim=capsule_dim,
            routing_iters=5
        )

        # ====================================================
        # FLAT FEATURE HEAD
        # ====================================================

        self.flat_head = nn.Sequential(
            nn.LayerNorm(self.n_branches * hidden_dim),
            nn.Linear(self.n_branches * hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResBlock(hidden_dim * 2, dropout)
        )

        # ====================================================
        # COARSE EMOTION HEAD
        # ====================================================

        self.coarse_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 96),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(96, num_coarse)
        )

        # ====================================================
        # AUXILIARY FINE HEAD
        # ====================================================

        self.aux_fine_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_fine)
        )

        # ====================================================
        # CAPSULE SCALING PARAMETER
        # ====================================================

        self.capsule_scale = nn.Parameter(
            torch.ones(num_fine)
        )

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):

        spec_end = self.spec_dim
        rel_end = spec_end + self.rel_dim
        pros_end = rel_end + self.pros_dim

        spec = x[:, :spec_end]
        rel = x[:, spec_end:rel_end]
        pros = x[:, rel_end:pros_end]

        s_h = self.spec_enc(spec).unsqueeze(1)
        r_h = self.rel_enc(rel).unsqueeze(1)
        p_h = self.pros_enc(pros).unsqueeze(1)

        branches = [s_h, r_h, p_h]

        if self.n_branches == 4:
            extra = x[:, pros_end:pros_end + self.extra_dim]
            x_h = self.extra_enc(extra).unsqueeze(1)
            branches.append(x_h)

        feats = torch.cat(branches, dim=1)

        fused = self.fusion(feats)

        primary = self.primary_proj(fused)

        primary = primary.view(
            x.size(0),
            self.n_branches * self.primary_caps_per_branch,
            self.capsule_dim
        )

        contextual, _ = self.capsule_context(
            primary,
            primary,
            primary
        )

        primary = self.context_norm(primary + contextual)

        emotion_caps = self.emotion_caps(primary)

        capsule_probs = torch.norm(
            emotion_caps,
            dim=-1
        ).clamp(max=1.0)

        pooled = self.flat_head(
            fused.flatten(1)
        )

        coarse_logits = self.coarse_head(pooled)

        aux_logits = self.aux_fine_head(pooled)

        capsule_logits = torch.logit(
            capsule_probs.clamp(1e-5, 1 - 1e-5)
        ) * self.capsule_scale

        fine_logits = aux_logits + capsule_logits

        return (
            emotion_caps,
            aux_logits,
            coarse_logits,
            fine_logits
        )
