import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull
from scipy.stats import entropy
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# HIERARCHICAL MARGIN LOSS
# ============================================================
#
# Combines:
#   - weighted cross-entropy on the fine (8-class) logits
#   - weighted cross-entropy on the coarse (4-class) logits, scaled by
#     coarse_weight
#   - a CapsNet-style margin loss over sigmoid(fine_logits), scaled by
#     capsule_weight -- fine_logits already embed the capsule-norm signal
#     (aux_logits + capsule_logits, see ARCANetV2.forward), so this
#     encourages well-separated per-class activation margins rather than
#     just correct argmax.
#
# fine_counts / coarse_counts are accepted for API compatibility with the
# training script (which computes them alongside the class weights) but
# are not used directly in the loss math -- the precomputed inverse-
# frequency weights (fine_weight / coarse_class_weight) already encode
# class-imbalance information.

class HierarchicalMarginLoss(nn.Module):

    def __init__(
        self,
        coarse_weight=0.3,
        label_smoothing=0.08,
        fine_weight=None,
        coarse_class_weight=None,
        fine_counts=None,
        coarse_counts=None,
        capsule_weight=0.40,
        margin_pos=0.9,
        margin_neg=0.1,
    ):
        super().__init__()

        self.coarse_weight = coarse_weight
        self.capsule_weight = capsule_weight
        self.label_smoothing = label_smoothing

        self.fine_weight = fine_weight
        self.coarse_class_weight = coarse_class_weight

        self.fine_counts = fine_counts
        self.coarse_counts = coarse_counts

        self.margin_pos = margin_pos
        self.margin_neg = margin_neg

    def forward(self, coarse_logits, fine_logits, coarse_labels, fine_labels):

        fine_ce = F.cross_entropy(
            fine_logits,
            fine_labels,
            weight=self.fine_weight,
            label_smoothing=self.label_smoothing,
        )

        coarse_ce = F.cross_entropy(
            coarse_logits,
            coarse_labels,
            weight=self.coarse_class_weight,
            label_smoothing=self.label_smoothing,
        )

        num_fine = fine_logits.size(1)
        target_onehot = F.one_hot(fine_labels, num_fine).float()
        probs = torch.sigmoid(fine_logits)

        pos_term = target_onehot * F.relu(self.margin_pos - probs) ** 2
        neg_term = 0.5 * (1 - target_onehot) * F.relu(probs - self.margin_neg) ** 2
        margin_loss = (pos_term + neg_term).sum(dim=1).mean()

        return fine_ce + self.coarse_weight * coarse_ce + self.capsule_weight * margin_loss


# ============================================================
# LABEL DEFINITIONS
# ============================================================

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

FINE_TO_IDX = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7,
}

FINE_TO_COARSE = {
    "neutral": 0,
    "calm": 0,
    "happy": 1,
    "surprised": 1,
    "sad": 2,
    "fearful": 2,
    "angry": 3,
    "disgust": 3,
}


# ============================================================
# SIGNAL HELPERS
# ============================================================

def get_tkeo(y):
    return y[1:-1] ** 2 - y[:-2] * y[2:]


# ============================================================
# VISIBILITY GRAPH FEATURES (12)
# ============================================================

def extract_visibility_graph_features(series, max_points=80):
    series = np.asarray(series, dtype=np.float32)

    if len(series) < 4:
        return np.zeros(12, dtype=np.float32)

    if len(series) > max_points:
        idx = np.linspace(0, len(series) - 1, max_points).astype(int)
        series = series[idx]

    series = (series - np.mean(series)) / (np.std(series) + 1e-6)

    n = len(series)

    degrees = np.zeros(n)
    edge_lengths = []
    slopes = []

    for i in range(n - 1):
        yi = series[i]

        for j in range(i + 1, n):
            yj = series[j]

            if j == i + 1:
                visible = True
            else:
                t = np.arange(i + 1, j)
                interp = yi + (yj - yi) * ((t - i) / (j - i))
                visible = np.all(series[i + 1 : j] < interp)

            if visible:
                degrees[i] += 1
                degrees[j] += 1
                edge_lengths.append(j - i)
                slopes.append((yj - yi) / (j - i))

    if len(edge_lengths) == 0:
        return np.zeros(12)

    edge_lengths = np.array(edge_lengths)
    slopes = np.array(slopes)

    density = (2 * len(edge_lengths)) / (n * (n - 1))

    hubs = degrees / (n - 1)

    return np.array(
        [
            np.mean(degrees),
            np.std(degrees),
            np.max(degrees),
            entropy(degrees + 1e-6),
            density,
            np.mean(edge_lengths),
            np.std(edge_lengths),
            np.max(edge_lengths),
            np.mean(np.abs(slopes)),
            np.std(slopes),
            np.max(hubs),
            np.mean(hubs > 0.25),
        ],
        dtype=np.float32,
    )


# ============================================================
# FEATURE EXTRACTION (361 FEATURES)
# ============================================================

def extract_enhanced_features(wav_path, sr=16000):
    try:
        y, sr = librosa.load(wav_path, sr=sr)

        if len(y) == 0:
            return None

        y = librosa.util.normalize(y)

        # MFCC (120)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)

        mfcc_feats = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(d1, axis=1),
                np.std(d1, axis=1),
                np.mean(d2, axis=1),
                np.std(d2, axis=1),
            ]
        )

        # MEL (128)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        log_mel = librosa.power_to_db(mel)
        mel_feats = np.concatenate([np.mean(log_mel, axis=1), np.std(log_mel, axis=1)])

        # CHROMA (24)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_feats = np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])

        # SPECTRAL CONTRAST (14)
        spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_con_feats = np.concatenate(
            [np.mean(spec_con, axis=1), np.std(spec_con, axis=1)]
        )

        # OTHER SPECTRAL (8)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)

        S, _ = librosa.magphase(librosa.stft(y))
        spec_ent = entropy(S, axis=0)

        spec_misc = np.array(
            [
                np.mean(spec_cent),
                np.std(spec_cent),
                np.mean(spec_bw),
                np.std(spec_bw),
                np.mean(spec_roll),
                np.std(spec_roll),
                np.mean(spec_ent),
                np.std(spec_ent),
            ]
        )

        # VOICE QUALITY (10)
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        f0_clean = f0[~np.isnan(f0)]

        if len(f0_clean) > 0:
            pitch_feats = [
                np.mean(f0_clean),
                np.std(f0_clean),
                np.max(f0_clean) - np.min(f0_clean),
                np.mean(voiced_flag),
            ]
            periods = 1 / f0_clean
            jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods)
        else:
            pitch_feats = [0, 0, 0, 0]
            jitter = 0

        autocorr = librosa.autocorrelate(y)
        hnr = 10 * np.log10(
            np.max(autocorr)
            / (np.sum(autocorr) - np.max(autocorr) + 1e-8)
        )
        hnr_feats = [hnr, 0]

        tkeo = get_tkeo(y)
        tkeo_feats = [np.mean(tkeo), np.std(tkeo)]

        rms = librosa.feature.rms(y=y)[0]

        shimmer = (
            np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-8)
            if len(rms) > 1
            else 0
        )

        voice_quality = np.concatenate(
            [pitch_feats, hnr_feats, tkeo_feats, [jitter, shimmer]]
        )

        # ENERGY + ZCR (4)
        zcr = librosa.feature.zero_crossing_rate(y)
        energy_feats = [
            np.mean(rms),
            np.std(rms),
            np.mean(zcr),
            np.std(zcr),
        ]

        # VISIBILITY GRAPH (12)
        vg_feats = extract_visibility_graph_features(rms)

        feats = np.concatenate(
            [
                mfcc_feats,
                mel_feats,
                spec_misc,
                chroma_feats,
                spec_con_feats,
                voice_quality,
                energy_feats,
                vg_feats,
            ]
        )

        return np.nan_to_num(feats)

    except Exception:
        return None


# ============================================================
# DATASET PIPELINE
# ============================================================

def process_file(args):
    path, emotion = args

    X = extract_enhanced_features(path)
    if X is None:
        return None
    return X, FINE_TO_IDX[emotion], FINE_TO_COARSE[emotion]


def load_dataset(dataset_root, cache_path="features_enhanced.pth"):
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return torch.load(cache_path)["features"]

    print("Extracting features...")

    files_to_process = []

    # ---- RAVDESS style (default) ----
    for subdir, _, files in os.walk(dataset_root):
        for file in sorted(files):
            if file.endswith(".wav"):
                parts = file.split("-")
                if len(parts) >= 7 and parts[1] == "01":
                    emotion = EMOTION_MAP.get(parts[2])
                    if emotion:
                        files_to_process.append(
                            (os.path.join(subdir, file), emotion)
                        )

    # ---- IEMOCAP style ----
    iemocap_candidates = [
        os.path.join(dataset_root, "IEMOCAP_full_unsegmentated"),
        os.path.join(dataset_root, "IEMOCAP_full_release"),
        os.path.join(dataset_root, "IEMOCAP"),
        dataset_root,
    ]

    iemocap_root = None
    for candidate in iemocap_candidates:
        if os.path.exists(candidate):
            # Check if it contains session directories
            subdirs = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]
            if any("Session" in d or "session" in d.lower() for d in subdirs):
                iemocap_root = candidate
                break

    if iemocap_root:
        for subdir, _, files in os.walk(iemocap_root):
            for file in sorted(files):
                if not file.endswith(".wav"):
                    continue

                wav_path = os.path.join(subdir, file)
                stem = os.path.splitext(file)[0]

                # Look for annotation file in multiple possible locations
                possible_locations = [
                    os.path.join(os.path.dirname(subdir), "label", f"{stem}.txt"),
                    os.path.join(os.path.dirname(os.path.dirname(subdir)), "label", f"{stem}.txt"),
                    os.path.join(iemocap_root, "label", f"{stem}.txt"),
                ]

                emotion = None
                for possible_annot in possible_locations:
                    if os.path.exists(possible_annot):
                        try:
                            with open(possible_annot, "r", encoding="utf-8") as f:
                                content = f.read().lower()
                                # Map IEMOCAP emotion codes to standard emotions
                                if "[ang]" in content or " ang " in content:
                                    emotion = "angry"
                                elif "[hap]" in content or " hap " in content or "[exc]" in content:
                                    emotion = "happy"
                                elif "[sad]" in content or " sad " in content:
                                    emotion = "sad"
                                elif "[neu]" in content or " neu " in content:
                                    emotion = "neutral"
                                elif "[fea]" in content or " fea " in content:
                                    emotion = "fearful"
                                elif "[dis]" in content or " dis " in content:
                                    emotion = "disgust"
                                elif "[sur]" in content or " sur " in content or "[fru]" in content:
                                    emotion = "surprised"

                                if emotion:
                                    break
                        except Exception:
                            continue

                if emotion:
                    files_to_process.append((wav_path, emotion))

    # ---- EMODB style (simple file‑based labels) ----
    emodb_root = dataset_root
    if os.path.exists(os.path.join(dataset_root, "wav")):
        emodb_root = dataset_root
    elif os.path.exists(os.path.join(dataset_root, "..", "wav")):
        emodb_root = os.path.dirname(dataset_root)

    label_map = {
        "neutral": "neutral",
        "calm": "calm",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fearful",
        "disgust": "disgust",
        "surprised": "surprised",
    }

    wav2emotion_path = os.path.join(emodb_root, "wav2emotion.txt")
    if os.path.exists(wav2emotion_path):
        with open(wav2emotion_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                wav_name = parts[0]
                emo_code = parts[1].lower()
                emo_name = label_map.get(emo_code, emo_code)
                wav_path = os.path.join(emodb_root, "wav", wav_name)
                if os.path.exists(wav_path):
                    files_to_process.append((wav_path, emo_name))
    else:
        # Fallback: parse filenames directly.
        # Standard EMODB format: [speaker(2)][text code(3)][emotion(1)][version(1)].wav
        # e.g. "03a01Fa.wav" -> speaker=03, text=a01, emotion=F, version=a
        # Emotion letter is the second-to-last character of the filename stem.
        wav_dir = os.path.join(emodb_root, "wav")
        if os.path.exists(wav_dir):
            emodb_emotion_map = {
                "w": "angry",       # Ärger (Wut)
                "l": "calm",        # Langeweile (boredom -> closest available class)
                "e": "disgust",     # Ekel
                "a": "fearful",     # Angst
                "f": "happy",       # Freude
                "t": "sad",         # Trauer
                "n": "neutral",     # Neutral
            }
            for file in os.listdir(wav_dir):
                if file.endswith(".wav"):
                    stem = os.path.splitext(file)[0]
                    if len(stem) < 2:
                        continue
                    emotion_char = stem[-2].lower()
                    emotion = emodb_emotion_map.get(emotion_char)
                    if emotion:
                        wav_path = os.path.join(wav_dir, file)
                        files_to_process.append((wav_path, emotion))

    dataset = []

    with Pool(min(4, os.cpu_count())) as pool:
        for result in pool.imap(process_file, files_to_process):
            if result is not None:
                dataset.append(result)

    torch.save({"features": dataset}, cache_path)
    return dataset


class SERDataset(Dataset):
    def __init__(self, data, mean=None, std=None):
        self.data = data
        Xs = np.array([x[0] for x in data])
        self.mean = mean if mean is not None else Xs.mean(axis=0)
        self.std = std if std is not None else Xs.std(axis=0) + 1e-6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, f, c = self.data[idx]
        X = (X - self.mean) / self.std
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(f),
            torch.tensor(c),
        )


# ============================================================
# BALANCED SAMPLER
# ============================================================

def make_weighted_sampler(dataset, num_classes=8):
    labels = np.array([x[1] for x in dataset.data])
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / counts
    sample_weights = weights[labels]
    return WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )


# ============================================================
# DATALOADER FACTORY
# ============================================================

def get_dataloaders(
    dataset_root,
    batch_size=64,
    cache_path="features_enhanced.pth",
    balanced_sampling=True,
):
    full_data = load_dataset(dataset_root, cache_path)

    train_data, test_data = train_test_split(
        full_data,
        test_size=0.2,
        stratify=[x[1] for x in full_data],
        random_state=42,
    )

    train_ds = SERDataset(train_data)

    test_ds = SERDataset(
        test_data, mean=train_ds.mean, std=train_ds.std
    )

    sampler = (
        make_weighted_sampler(train_ds)
        if balanced_sampling
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, len(full_data)