"""
Advanced feature families for the ARCA-Net ablation study.

Each extractor takes a raw waveform `y` (already loaded + amplitude-normalized,
matching utils.extract_enhanced_features' preprocessing) and the sample rate `sr`,
and returns a FIXED-LENGTH numpy vector of summary statistics. Every function
aggregates (mean/std/entropy/etc.) rather than emitting raw positional values,
so output length never depends on clip duration or how many components an
intermediate decomposition happens to find.

Families (per the feature-engineering brief):
  1. koopman      - Nonlinear speech dynamics via Koopman/DMD eigen-analysis
  2. gsp          - Graph Signal Processing features on the speech envelope
  3. tda          - Topological Data Analysis / persistent-homology features
  4. hamiltonian  - Hamiltonian/Lagrangian energy-dynamics features
  5. rqa          - Recurrence Quantification Analysis
  6. visibility_graph_v2 - Visibility-graph network-theoretic features
  7. info_geometry - Information-geometry features over frame distributions

All are self-contained (numpy/scipy/networkx/librosa only) to avoid fragile
external TDA/RQA dependencies (ripser/gudhi are not installed in this env).
"""

import numpy as np
import librosa
import networkx as nx
from scipy.stats import entropy as shannon_entropy
from scipy.linalg import eigh


# ============================================================
# SHARED HELPERS
# ============================================================

def _safe_entropy(values, base=None):
    values = np.asarray(values, dtype=np.float64)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    p = values / values.sum()
    return float(shannon_entropy(p, base=base))


def get_envelope(y, sr, max_points=120, hop_length=256, frame_length=1024):
    """Downsampled, z-scored RMS energy envelope used as the base 1-D signal."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    if len(rms) > max_points:
        idx = np.linspace(0, len(rms) - 1, max_points).astype(int)
        rms = rms[idx]

    rms = rms.astype(np.float64)
    std = rms.std()
    if std < 1e-8:
        return rms - rms.mean()
    return (rms - rms.mean()) / std


def delay_embed(x, m=3, tau=2):
    """Takens delay embedding: returns points of shape (N - (m-1)*tau, m)."""
    n = len(x) - (m - 1) * tau
    if n <= 1:
        return np.zeros((0, m))
    return np.stack([x[i * tau: i * tau + n] for i in range(m)], axis=1)


def _union_find_components(n, edges_sorted):
    """edges_sorted: list of (dist, i, j) ascending. Returns MST merge distances
    (H0 lifetimes) and the number of connected components as edges are added."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    merge_dists = []
    for dist, i, j in edges_sorted:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            merge_dists.append(dist)
    return merge_dists


def _pairwise_dist(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


# ============================================================
# 1. KOOPMAN / DYNAMIC MODE DECOMPOSITION FEATURES
# ============================================================

def extract_koopman_features(y, sr, max_points=120, delay_dim=8, max_rank=6):
    e = get_envelope(y, sr, max_points=max_points)
    n = len(e)

    if n < delay_dim + 2:
        return np.zeros(10, dtype=np.float32)

    # Hankel delay-embedding snapshot matrix
    cols = n - delay_dim
    X = np.stack([e[i:i + cols] for i in range(delay_dim)], axis=0)
    X1, X2 = X[:, :-1], X[:, 1:]

    try:
        U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(10, dtype=np.float32)

    r = min(max_rank, np.sum(S > 1e-8))
    if r < 1:
        return np.zeros(10, dtype=np.float32)

    Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
    A_tilde = Ur.T @ X2 @ Vhr.T @ np.diag(1.0 / Sr)

    try:
        eigvals, eigvecs = np.linalg.eig(A_tilde)
    except np.linalg.LinAlgError:
        return np.zeros(10, dtype=np.float32)

    mags = np.abs(eigvals)
    phases = np.angle(eigvals)

    # DMD mode energies: norm of each mode's contribution in the data space
    try:
        Phi = X2 @ Vhr.T @ np.diag(1.0 / Sr) @ eigvecs
        mode_energy = np.linalg.norm(Phi, axis=0)
    except Exception:
        mode_energy = Sr[:r]

    stability = mags - 1.0  # 0 = marginally stable, <0 decaying, >0 growing

    feats = np.array([
        mags.mean(), mags.std(), mags.max(),
        phases.mean(), phases.std(),
        stability.mean(), stability.std(),
        _safe_entropy(mode_energy),
        mode_energy.mean(), mode_energy.std(),
    ], dtype=np.float32)

    return np.nan_to_num(feats)


# ============================================================
# 2. GRAPH SIGNAL PROCESSING (GSP) FEATURES
# ============================================================

def extract_gsp_features(y, sr, max_points=100, window=3):
    e = get_envelope(y, sr, max_points=max_points)
    n = len(e)

    if n < 4:
        return np.zeros(8, dtype=np.float32)

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if i == j:
                continue
            w = np.exp(-((e[i] - e[j]) ** 2) / 2.0)
            W[i, j] = w
            W[j, i] = w

    D = np.diag(W.sum(axis=1))
    L = D - W

    try:
        eigvals = eigh(L, eigvals_only=True)
    except np.linalg.LinAlgError:
        return np.zeros(8, dtype=np.float32)

    eigvals = np.clip(eigvals, 0, None)
    nonzero = eigvals[eigvals > 1e-6]
    algebraic_connectivity = nonzero.min() if nonzero.size > 0 else 0.0

    x = e
    denom = float(x @ x) + 1e-8
    smoothness = float(x @ L @ x) / denom

    edge_mask = W > 0
    n_edges = edge_mask.sum() / 2.0
    total_variation = (
        (W * (e[:, None] - e[None, :]) ** 2).sum() / 2.0
    ) / max(n_edges, 1.0)

    feats = np.array([
        eigvals.mean(), eigvals.std(), eigvals.max(),
        algebraic_connectivity,
        eigvals.max() - eigvals.min(),
        _safe_entropy(eigvals),
        smoothness,
        total_variation,
    ], dtype=np.float32)

    return np.nan_to_num(feats)


# ============================================================
# 3. TOPOLOGICAL DATA ANALYSIS (TDA) FEATURES
# ============================================================

def extract_tda_features(y, sr, max_points=60, m=3, tau=2):
    e = get_envelope(y, sr, max_points=max_points + 2 * tau)
    points = delay_embed(e, m=m, tau=tau)

    if points.shape[0] < 4:
        return np.zeros(8, dtype=np.float32)

    n = points.shape[0]
    D = _pairwise_dist(points)
    iu = np.triu_indices(n, k=1)
    dists = D[iu]

    if dists.size == 0:
        return np.zeros(8, dtype=np.float32)

    eps1, eps2 = np.percentile(dists, [25, 50])

    edges_sorted = sorted(zip(dists, iu[0], iu[1]), key=lambda t: t[0])
    merge_dists = _union_find_components(n, edges_sorted)
    merge_dists = np.array(merge_dists) if merge_dists else np.zeros(1)

    def betti_at(eps):
        adj = D <= eps
        np.fill_diagonal(adj, False)
        n_edges = adj.sum() / 2.0

        g = nx.Graph()
        g.add_nodes_from(range(n))
        ii, jj = np.where(np.triu(adj, k=1))
        g.add_edges_from(zip(ii, jj))
        n_components = nx.number_connected_components(g)

        betti0 = n_components
        betti1 = max(0, int(n_edges) - n + n_components)
        return betti0 / n, betti1 / n

    betti0_e1, betti1_e1 = betti_at(eps1)
    betti0_e2, betti1_e2 = betti_at(eps2)

    feats = np.array([
        betti0_e1, betti0_e2,
        betti1_e1, betti1_e2,
        _safe_entropy(merge_dists),
        merge_dists.mean(), merge_dists.std(), merge_dists.max(),
    ], dtype=np.float32)

    return np.nan_to_num(feats)


# ============================================================
# 4. HAMILTONIAN / LAGRANGIAN DYNAMICS FEATURES
# ============================================================

def extract_hamiltonian_features(y, sr, max_points=120):
    x = get_envelope(y, sr, max_points=max_points)
    n = len(x)

    if n < 4:
        return np.zeros(8, dtype=np.float32)

    v = np.gradient(x)
    H = 0.5 * v ** 2 + 0.5 * x ** 2

    dHdt = np.gradient(H)

    t = np.arange(n, dtype=np.float64)
    log_H = np.log(H + 1e-6)
    try:
        dissipation_coef = float(np.polyfit(t, log_H, 1)[0])
    except Exception:
        dissipation_coef = 0.0

    theta = np.arctan2(v, x)

    if H.std() > 1e-8 and theta.std() > 1e-8:
        coupling = float(np.corrcoef(theta, H)[0, 1])
    else:
        coupling = 0.0

    feats = np.array([
        H.mean(), H.std(), H.max(),
        np.abs(dHdt).mean(), dHdt.std(),
        dissipation_coef,
        coupling,
        theta.std(),
    ], dtype=np.float32)

    return np.nan_to_num(feats)


# ============================================================
# 5. RECURRENCE QUANTIFICATION ANALYSIS (RQA)
# ============================================================

def extract_rqa_features(y, sr, max_points=100, m=3, tau=2, min_line=2):
    e = get_envelope(y, sr, max_points=max_points + 2 * tau)
    points = delay_embed(e, m=m, tau=tau)

    if points.shape[0] < min_line + 2:
        return np.zeros(5, dtype=np.float32)

    n = points.shape[0]
    D = _pairwise_dist(points)

    max_d = D.max()
    if max_d < 1e-8:
        return np.zeros(5, dtype=np.float32)

    eps = 0.1 * max_d
    R = (D <= eps).astype(np.int8)
    np.fill_diagonal(R, 0)

    total_points = R.sum()
    if total_points == 0:
        return np.zeros(5, dtype=np.float32)

    rr = total_points / (n * n)

    # Diagonal lines (excluding the main diagonal)
    diag_lengths = []
    for k in range(-(n - 1), n):
        if k == 0:
            continue
        diag = np.diagonal(R, offset=k)
        run = 0
        for val in diag:
            if val:
                run += 1
            else:
                if run >= min_line:
                    diag_lengths.append(run)
                run = 0
        if run >= min_line:
            diag_lengths.append(run)

    diag_lengths = np.array(diag_lengths) if diag_lengths else np.zeros(0)
    det = (diag_lengths.sum() / total_points) if diag_lengths.size > 0 else 0.0
    diag_entropy = _safe_entropy(diag_lengths) if diag_lengths.size > 0 else 0.0

    # Vertical lines
    vert_lengths = []
    for col in range(n):
        run = 0
        for row in range(n):
            if R[row, col]:
                run += 1
            else:
                if run >= min_line:
                    vert_lengths.append(run)
                run = 0
        if run >= min_line:
            vert_lengths.append(run)

    vert_lengths = np.array(vert_lengths) if vert_lengths else np.zeros(0)
    lam = (vert_lengths.sum() / total_points) if vert_lengths.size > 0 else 0.0
    tt = vert_lengths.mean() if vert_lengths.size > 0 else 0.0

    feats = np.array([rr, det, lam, tt, diag_entropy], dtype=np.float32)
    return np.nan_to_num(feats)


# ============================================================
# 6. VISIBILITY GRAPH V2 (NETWORK-THEORETIC) FEATURES
# ============================================================

def extract_visibility_graph_v2_features(y, sr, max_points=70):
    e = get_envelope(y, sr, max_points=max_points)
    n = len(e)

    if n < 4:
        return np.zeros(6, dtype=np.float32)

    g = nx.Graph()
    g.add_nodes_from(range(n))

    for i in range(n - 1):
        g.add_edge(i, i + 1)
        yi = e[i]
        for j in range(i + 2, n):
            yj = e[j]
            t = np.arange(i + 1, j)
            interp = yi + (yj - yi) * ((t - i) / (j - i))
            if np.all(e[i + 1:j] < interp):
                g.add_edge(i, j)

    degrees = np.array([d for _, d in g.degree()], dtype=np.float64)

    clustering = nx.average_clustering(g)

    if nx.is_connected(g):
        avg_path_len = nx.average_shortest_path_length(g)
    else:
        largest_cc = max(nx.connected_components(g), key=len)
        sub = g.subgraph(largest_cc)
        avg_path_len = (
            nx.average_shortest_path_length(sub) if len(sub) > 1 else 0.0
        )

    try:
        norm_lap_eigs = nx.normalized_laplacian_spectrum(g)
        network_entropy = _safe_entropy(np.abs(norm_lap_eigs))
    except Exception:
        network_entropy = 0.0

    feats = np.array([
        degrees.mean(), degrees.std(), _safe_entropy(degrees),
        clustering, avg_path_len, network_entropy,
    ], dtype=np.float32)

    return np.nan_to_num(feats)


# ============================================================
# 7. INFORMATION GEOMETRY FEATURES
# ============================================================

def extract_information_geometry_features(y, sr, n_mels=20, max_frames=50):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel = np.clip(mel, 1e-10, None)

    if mel.shape[1] > max_frames:
        idx = np.linspace(0, mel.shape[1] - 1, max_frames).astype(int)
        mel = mel[:, idx]

    # Normalize each frame into a probability distribution over mel bins
    P = mel / mel.sum(axis=0, keepdims=True)
    T = P.shape[1]

    if T < 3:
        return np.zeros(9, dtype=np.float32)

    def fisher_rao_dist(p, q):
        bc = np.clip(np.sum(np.sqrt(p * q)), 0.0, 1.0)
        return 2.0 * np.arccos(bc)

    def kl_div(p, q):
        return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))

    def renyi_div(p, q, alpha=0.5):
        s = np.sum((p ** alpha) * (q ** (1 - alpha)))
        s = np.clip(s, 1e-12, None)
        return float(np.log(s) / (alpha - 1))

    fr_dists = np.array([
        fisher_rao_dist(P[:, t], P[:, t + 1]) for t in range(T - 1)
    ])
    kl_vals = np.array([
        kl_div(P[:, t], P[:, t + 1]) for t in range(T - 1)
    ])
    renyi_vals = np.array([
        renyi_div(P[:, t], P[:, t + 1]) for t in range(T - 1)
    ])

    curvature = np.abs(fr_dists[2:] - 2 * fr_dists[1:-1] + fr_dists[:-2]) \
        if len(fr_dists) >= 3 else np.zeros(1)

    quarter = max(1, T // 4)
    early_dist = P[:, :quarter].mean(axis=1)
    late_dist = P[:, -quarter:].mean(axis=1)
    early_dist = early_dist / early_dist.sum()
    late_dist = late_dist / late_dist.sum()
    early_late_dist = fisher_rao_dist(early_dist, late_dist)

    feats = np.array([
        fr_dists.mean(), fr_dists.std(),
        fr_dists.max() - fr_dists.min(),
        early_late_dist,
        curvature.mean(), curvature.std(),
        kl_vals.mean(), kl_vals.std(),
        renyi_vals.mean(),
    ], dtype=np.float32)

    return np.nan_to_num(feats)


# ============================================================
# REGISTRY
# ============================================================

FEATURE_FAMILIES = {
    "koopman": (extract_koopman_features, 10),
    "gsp": (extract_gsp_features, 8),
    "tda": (extract_tda_features, 8),
    "hamiltonian": (extract_hamiltonian_features, 8),
    "rqa": (extract_rqa_features, 5),
    "visibility_graph_v2": (extract_visibility_graph_v2_features, 6),
    "info_geometry": (extract_information_geometry_features, 9),
}


def extract_all_families(y, sr):
    """Returns a dict {family_name: np.ndarray} for every registered family."""
    out = {}
    for name, (func, dim) in FEATURE_FAMILIES.items():
        try:
            vec = func(y, sr)
            if vec.shape[0] != dim:
                vec = np.zeros(dim, dtype=np.float32)
        except Exception:
            vec = np.zeros(dim, dtype=np.float32)
        out[name] = vec
    return out
