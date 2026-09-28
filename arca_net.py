import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CAPSULE SQUASH FUNCTION
# ============================================================

def squash(v, dim=-1):

    norm_sq = (v ** 2).sum(dim=dim, keepdim=True)

    scale = norm_sq / (1.0 + norm_sq)

    return scale * v / torch.sqrt(norm_sq + 1e-8)


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResBlock(nn.Module):

    def __init__(self, dim, dropout=0.35):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):

        return x + self.net(x)


# ============================================================
# ATTENTIVE ROUTING CAPSULE LAYER
# ============================================================

class AttentiveRoutingCapsuleLayer(nn.Module):

    def __init__(
        self,
        in_caps,
        out_caps,
        in_dim,
        out_dim,
        routing_iters=5
    ):

        super().__init__()

        self.routing_iters = routing_iters

        self.route_weights = nn.Parameter(
            torch.randn(1, in_caps, out_caps, out_dim, in_dim) * 0.02
        )

        self.attn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.SiLU(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x):

        batch_size = x.size(0)

        prior = torch.matmul(
            self.route_weights.expand(batch_size, -1, -1, -1, -1),
            x.unsqueeze(2).unsqueeze(-1)
        ).squeeze(-1)

        capsule_importance = torch.softmax(
            self.attn(x).squeeze(-1),
            dim=1
        ).unsqueeze(-1)

        logits = torch.zeros(
            x.size(0),
            x.size(1),
            prior.size(2),
            device=x.device
        )

        for i in range(self.routing_iters):

            route = torch.softmax(logits, dim=-1) * capsule_importance

            out = squash(
                (route.unsqueeze(-1) * prior).sum(dim=1)
            )

            if i < self.routing_iters - 1:

                logits = logits + (
                    prior * out.unsqueeze(1)
                ).sum(dim=-1)

        return out


# ============================================================
# ARCA-NET V2
# ============================================================

class ARCANetV2(nn.Module):

    """
    ARCA-Net v2
    Input = 361 handcrafted acoustic features

    Feature grouping:

    Spectral  = 256
    Relational = 38
    Prosodic   = 67
    """

    def __init__(
        self,
        num_coarse=4,
        num_fine=8,
        spec_dim=256,
        rel_dim=38,
        pros_dim=26,
        hidden_dim=128,
        capsule_dim=24,
        primary_caps_per_branch=4,
        dropout=0.35
    ):

        super().__init__()

        self.spec_dim = spec_dim
        self.rel_dim = rel_dim
        self.pros_dim = pros_dim
        self.num_fine = num_fine
        self.capsule_dim = capsule_dim
        self.primary_caps_per_branch = primary_caps_per_branch

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
            in_caps=3 * primary_caps_per_branch,
            out_caps=num_fine,
            in_dim=capsule_dim,
            out_dim=capsule_dim,
            routing_iters=5
        )

        # ====================================================
        # FLAT FEATURE HEAD
        # ====================================================

        self.flat_head = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim * 2),
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

        spec = x[:, :spec_end]
        rel = x[:, spec_end:rel_end]
        pros = x[:, rel_end:rel_end + self.pros_dim]

        s_h = self.spec_enc(spec).unsqueeze(1)
        r_h = self.rel_enc(rel).unsqueeze(1)
        p_h = self.pros_enc(pros).unsqueeze(1)

        feats = torch.cat([s_h, r_h, p_h], dim=1)

        fused = self.fusion(feats)

        primary = self.primary_proj(fused)

        primary = primary.view(
            x.size(0),
            3 * self.primary_caps_per_branch,
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