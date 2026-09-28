"""
Feature-ablation study for ARCA-Net v2 on RAVDESS.

Tests 7 advanced feature families (see advanced_features.py) individually
and in every pairwise combination (28 configs total) against a baseline
(the existing 320-dim handcrafted feature set with no additions), using a
reduced single-seed training budget per config for tractability, then
produces a ranked comparison report.

Design notes (see conversation for the full rationale):
  - The SAME train/test split (indices) is reused across every config, so
    accuracy differences reflect the feature content, not split variance.
  - Each of the 7 families' feature vectors is extracted ONCE per file and
    cached; the 29 configs are built by concatenating cached vectors, so
    the expensive extraction pass happens only once.
  - The model is ARCANetV2Ablation, which is architecturally identical to
    the production ARCANetV2 when no extra features are added (baseline),
    and gains one additional encoder branch when extra features are
    present.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from itertools import combinations
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import librosa

from utils import (
    extract_enhanced_features,
    HierarchicalMarginLoss,
    FINE_TO_IDX,
    FINE_TO_COARSE,
)
from advanced_features import FEATURE_FAMILIES, extract_all_families
from arca_net_ablation import ARCANetV2Ablation


# ============================================================
# CONFIG
# ============================================================

# Resolve paths relative to this Python file, not the terminal's current directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SPEECH_DIR = os.path.join(BASE_DIR, "speech", "RAVDESS")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "ablation_study")
CACHE_PATH = os.path.join(BASE_DIR, "ablation_features_cache.pth")

SEED = 42
EPOCHS = 35
BATCH_SIZE = 64
LR = 8e-4
TEST_SIZE = 0.2

FAMILY_NAMES = list(FEATURE_FAMILIES.keys())


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# STEP 1: FILE DISCOVERY (RAVDESS)
# ============================================================

def load_ravdess_files(data_dir):
    """Recursively find RAVDESS speech WAV files and decode emotions."""
    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    data_dir = os.path.abspath(data_dir)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"\nRAVDESS directory does not exist:\n{data_dir}\n"
            "Put the RAVDESS dataset in a folder named speech/RAVDESS "
            "relative to this script, or change SPEECH_DIR."
        )

    files = []
    total_wav = 0
    rejected = 0

    for subdir, _, fnames in os.walk(data_dir):
        for fname in sorted(fnames):
            if not fname.lower().endswith(".wav"):
                continue

            total_wav += 1
            parts = fname.split("-")

            # RAVDESS filename:
            # actor-emotion-intensity-statement-repetition-actor...
            # We keep speech files (second field == 01).
            if len(parts) < 7 or parts[1] != "01":
                rejected += 1
                continue

            emotion = emotion_map.get(parts[2])
            if emotion is None:
                rejected += 1
                continue

            files.append((os.path.join(subdir, fname), emotion))

    print(f"RAVDESS directory : {data_dir}")
    print(f"Total WAV files   : {total_wav}")
    print(f"Speech WAV files  : {len(files)}")

    if not files:
        raise RuntimeError(
            f"\nNo valid RAVDESS speech files were found in:\n{data_dir}\n"
            "Check the dataset path and folder structure."
        )

    return files


# ============================================================
# STEP 2: FEATURE EXTRACTION + CACHING
# ============================================================

def build_feature_cache():
    # Load cache only if it is valid and non-empty.
    if os.path.exists(CACHE_PATH):
        try:
            print(f"Loading cached ablation features from {CACHE_PATH}")
            cached = torch.load(CACHE_PATH, weights_only=False)

            if isinstance(cached, list) and len(cached) > 0:
                required = {"core", "families", "fine", "coarse"}
                if required.issubset(cached[0].keys()):
                    print(f"Loaded {len(cached)} cached samples.")
                    return cached

            print("Cache is empty or invalid. Rebuilding it...")
        except Exception as e:
            print(f"Could not load cache ({e}). Rebuilding it...")

    print("\nExtracting core (320-dim) + advanced-family features...")
    files = load_ravdess_files(SPEECH_DIR)
    print(f"Found {len(files)} candidate files.")

    cache = []
    t0 = time.time()
    skipped_core = 0
    skipped_family = 0

    for i, (path, emotion) in enumerate(files):
        try:
            if emotion not in FINE_TO_IDX:
                continue

            core = extract_enhanced_features(path)
            if core is None:
                skipped_core += 1
                continue

            y, sr = librosa.load(path, sr=16000)
            if y is None or len(y) == 0:
                skipped_family += 1
                continue

            y = librosa.util.normalize(y)
            family_feats = extract_all_families(y, sr)

            if not isinstance(family_feats, dict):
                skipped_family += 1
                continue

            missing = [fam for fam in FAMILY_NAMES if fam not in family_feats]
            if missing:
                print(f"\nWarning: {os.path.basename(path)} missing families: {missing}")
                skipped_family += 1
                continue

            cache.append({
                "path": path,
                "core": np.asarray(core, dtype=np.float32),
                "families": {
                    k: np.asarray(v, dtype=np.float32)
                    for k, v in family_feats.items()
                },
                "fine": FINE_TO_IDX[emotion],
                "coarse": FINE_TO_COARSE[emotion],
            })

        except Exception as e:
            print(f"\nSkipping failed file: {path}")
            print(f"Reason: {type(e).__name__}: {e}")
            continue

        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(files) - i - 1) / max(rate, 1e-6)
            print(
                f"  Processed {i+1}/{len(files)} | "
                f"cached={len(cache)} | ETA={eta:.0f}s"
            )

    print(f"\nExtracted {len(cache)} usable samples.")
    print(f"Skipped core extraction: {skipped_core}")
    print(f"Skipped family extraction: {skipped_family}")

    if len(cache) == 0:
        raise RuntimeError(
            "\nFeature extraction produced ZERO samples.\n"
            "The RAVDESS path is valid, but feature extraction returned no usable data. "
            "Check extract_enhanced_features() and extract_all_families()."
        )

    # Verify all required arrays have compatible lengths.
    core_dim = cache[0]["core"].shape[0]
    for item in cache:
        if item["core"].ndim != 1 or item["core"].shape[0] != core_dim:
            raise RuntimeError("Inconsistent core feature dimensions in cache.")

        for fam in FAMILY_NAMES:
            if item["families"][fam].ndim != 1:
                raise RuntimeError(
                    f"Feature family '{fam}' is not a 1-D vector."
                )

    torch.save(cache, CACHE_PATH)
    print(f"Cached to: {CACHE_PATH}")

    return cache


# ============================================================
# STEP 3: DATASET
# ============================================================

class AblationDataset(Dataset):
    def __init__(self, X, fine, coarse, mean=None, std=None):
        self.mean = mean if mean is not None else X.mean(axis=0)
        self.std = std if std is not None else X.std(axis=0) + 1e-6
        self.X = (X - self.mean) / self.std
        self.fine = fine
        self.coarse = coarse

    def __len__(self):
        return len(self.fine)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.fine[idx]),
            torch.tensor(self.coarse[idx]),
        )


def make_weighted_sampler(fine_labels, num_classes=8):
    counts = np.bincount(fine_labels, minlength=num_classes)
    weights = 1.0 / np.clip(counts, 1, None)
    sample_weights = weights[fine_labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def class_weight_tensor(labels, num_classes, device):
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights.to(device), counts


# ============================================================
# STEP 4: TRAIN / EVAL ONE CONFIG
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    correct_fine = correct_coarse = total = 0

    with torch.no_grad():
        for X, f, c in loader:
            X, f, c = X.to(device), f.to(device), c.to(device)
            _, _, coarse_logits, fine_logits = model(X)

            fine_preds = fine_logits.argmax(1)
            coarse_preds = coarse_logits.argmax(1)

            correct_fine += (fine_preds == f).sum().item()
            correct_coarse += (coarse_preds == c).sum().item()
            total += f.size(0)

    return correct_fine / total, correct_coarse / total


def run_config(config_name, family_list, cache, train_idx, test_idx, device):
    core = np.stack([c["core"] for c in cache])
    fine = np.array([c["fine"] for c in cache])
    coarse = np.array([c["coarse"] for c in cache])

    blocks = [core]
    for fam in family_list:
        blocks.append(np.stack([c["families"][fam] for c in cache]))

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    extra_dim = X.shape[1] - core.shape[1]

    X_train, X_test = X[train_idx], X[test_idx]
    fine_train, fine_test = fine[train_idx], fine[test_idx]
    coarse_train, coarse_test = coarse[train_idx], coarse[test_idx]

    train_ds = AblationDataset(X_train, fine_train, coarse_train)
    test_ds = AblationDataset(X_test, fine_test, coarse_test, mean=train_ds.mean, std=train_ds.std)

    sampler = make_weighted_sampler(fine_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    fine_weight, fine_counts = class_weight_tensor(fine_train, 8, device)
    coarse_weight, coarse_counts = class_weight_tensor(coarse_train, 4, device)

    criterion = HierarchicalMarginLoss(
        coarse_weight=0.3,
        label_smoothing=0.08,
        fine_weight=fine_weight,
        coarse_class_weight=coarse_weight,
        fine_counts=fine_counts,
        coarse_counts=coarse_counts,
        capsule_weight=0.40,
    )

    set_seed(SEED)
    model = ARCANetV2Ablation(extra_dim=extra_dim).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    best_fine_acc = 0.0
    best_coarse_acc = 0.0

    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for X_b, f_b, c_b in train_loader:
            X_b, f_b, c_b = X_b.to(device), f_b.to(device), c_b.to(device)

            optimizer.zero_grad()
            _, _, cp, fp = model(X_b)
            loss = criterion(cp, fp, c_b, f_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        fine_acc, coarse_acc = evaluate(model, test_loader, device)

        if fine_acc > best_fine_acc:
            best_fine_acc = fine_acc
            best_coarse_acc = coarse_acc

    elapsed = time.time() - t0

    return {
        "config": config_name,
        "families": ",".join(family_list) if family_list else "(baseline)",
        "n_families": len(family_list),
        "extra_dim": extra_dim,
        "fine_accuracy": best_fine_acc,
        "coarse_accuracy": best_coarse_acc,
        "train_time_sec": round(elapsed, 1),
    }


# ============================================================
# STEP 5: MAIN
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("ARCA-Net v2 Feature Ablation Study")
    print("=" * 70)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nProject directory:", BASE_DIR)
    print("RAVDESS directory:", SPEECH_DIR)
    print("Results directory:", RESULTS_DIR)
    print("Cache:", CACHE_PATH)

    if not os.path.isdir(SPEECH_DIR):
        raise FileNotFoundError(
            f"\nDataset not found:\n{SPEECH_DIR}\n\n"
            "Expected structure:\n"
            "project_folder/\n"
            "  ablation_study.py\n"
            "  speech/\n"
            "    RAVDESS/\n"
            "      Actor_01/\n"
            "      Actor_02/\n"
            "      ...\n"
        )

    cache = build_feature_cache()

    if len(cache) < 2:
        raise RuntimeError(
            f"Only {len(cache)} sample(s) are available. "
            "At least two samples are required for train/test splitting."
        )

    fine_labels = np.array([c["fine"] for c in cache])

    # Check class distribution before stratification.
    unique, counts = np.unique(fine_labels, return_counts=True)
    print("\nFine-class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  class {cls}: {count}")

    if len(unique) < 2:
        raise RuntimeError(
            "Only one emotion class was found. Stratified train/test split "
            "requires at least two classes."
        )

    if np.min(counts) < 2:
        raise RuntimeError(
            "At least one emotion class has fewer than 2 samples, so a "
            "stratified train/test split cannot be created."
        )

    all_idx = np.arange(len(cache))

    # Calculate expected split sizes and make sure the training set can
    # contain every class.
    test_count = max(1, int(np.ceil(len(all_idx) * TEST_SIZE)))
    train_count = len(all_idx) - test_count

    if train_count < len(unique):
        raise RuntimeError(
            f"Not enough samples for stratified split: "
            f"{len(all_idx)} total, {len(unique)} classes, "
            f"{train_count} training samples."
        )

    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        stratify=fine_labels,
        random_state=SEED,
    )

    print(f"\nTotal samples: {len(all_idx)}")
    print(f"Train: {len(train_idx)}")
    print(f"Test : {len(test_idx)}")

    configs = [("baseline", [])]
    configs += [(fam, [fam]) for fam in FAMILY_NAMES]
    configs += [
        (f"{a}+{b}", [a, b])
        for a, b in combinations(FAMILY_NAMES, 2)
    ]

    print(f"\nTotal configs to run: {len(configs)}")

    results_csv = os.path.join(RESULTS_DIR, "ablation_results.csv")
    results = []

    if os.path.exists(results_csv):
        try:
            done_df = pd.read_csv(results_csv)
            if "config" in done_df.columns:
                results = done_df.to_dict("records")
                done_names = set(done_df["config"].astype(str))
                configs = [
                    c for c in configs
                    if c[0] not in done_names
                ]
                print(
                    f"Resuming: {len(done_names)} configs already completed, "
                    f"{len(configs)} remaining."
                )
        except Exception as e:
            print(f"Could not read previous results ({e}). Starting fresh.")

    if not configs:
        print("\nAll configurations are already complete.")
    else:
        for i, (config_name, family_list) in enumerate(configs, start=1):
            print("\n" + "=" * 70)
            print(f"[{i}/{len(configs)}] Running config: {config_name}")
            print("=" * 70)

            try:
                result = run_config(
                    config_name,
                    family_list,
                    cache,
                    train_idx,
                    test_idx,
                    device,
                )

                results.append(result)

                print(
                    f"Fine Acc: {result['fine_accuracy']:.4f} | "
                    f"Coarse Acc: {result['coarse_accuracy']:.4f} | "
                    f"Time: {result['train_time_sec']}s"
                )

                pd.DataFrame(results).to_csv(
                    results_csv,
                    index=False,
                )

            except Exception as e:
                print(
                    f"\nERROR while running config '{config_name}': "
                    f"{type(e).__name__}: {e}"
                )
                print("Skipping this configuration and continuing.")
                continue

    # Final report.
    if results:
        df = pd.DataFrame(results)

        if "fine_accuracy" in df.columns:
            df = df.sort_values(
                "fine_accuracy",
                ascending=False,
            )

        df.to_csv(results_csv, index=False)

        print("\n" + "=" * 70)
        print("ABLATION STUDY COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {results_csv}")

        print("\nTop configurations by fine accuracy:")
        print(
            df[
                [
                    "config",
                    "families",
                    "extra_dim",
                    "fine_accuracy",
                    "coarse_accuracy",
                ]
            ].head(10).to_string(index=False)
        )
    else:
        print("\nNo configurations completed successfully.")


if __name__ == "__main__":
    main()
