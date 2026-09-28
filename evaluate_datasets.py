import torch
import torch.nn.functional as F
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pathlib import Path

from arca_net import ARCANetV2
from utils import SERDataset, extract_enhanced_features, FINE_TO_IDX, FINE_TO_COARSE
from torch.utils.data import DataLoader


# ============================================================
# DATASET LOADERS FOR DIFFERENT DATASETS
# ============================================================

def load_ravdess_dataset(data_dir):
    """Load RAVDESS dataset."""
    print("\nLoading RAVDESS dataset...")
    files_to_process = []

    for subdir, _, files in os.walk(data_dir):
        for file in sorted(files):
            if file.endswith(".wav"):
                parts = file.split("-")
                if len(parts) >= 7 and parts[1] == "01":
                    emotion_code = parts[2]
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
                    emotion = emotion_map.get(emotion_code)
                    if emotion:
                        files_to_process.append((os.path.join(subdir, file), emotion))

    return files_to_process


def load_iemocap_dataset(data_dir):
    """Load IEMOCAP dataset."""
    print("\nLoading IEMOCAP dataset...")
    files_to_process = []

    # ---- CSV-based layout (training_data.csv: file_id, emotion, audio_path) ----
    csv_path = os.path.join(data_dir, "training_data.csv")
    if os.path.exists(csv_path):
        csv_emotion_map = {
            "anger": "angry",
            "frustration": "angry",
            "excited": "happy",
            "happiness": "happy",
            "sadness": "sad",
            "neutral": "neutral",
            "fear": "fearful",
            "disgust": "disgust",
            "surprise": "surprised",
        }

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            emotion = csv_emotion_map.get(str(row["emotion"]).strip().lower())
            if not emotion:
                continue

            # audio_path in the CSV is a Kaggle-container path
            # (/kaggle/working/audio_files/...) -- remap to the local data_dir
            raw_path = str(row["audio_path"])
            marker = "audio_files"
            idx = raw_path.replace("\\", "/").find(marker)
            if idx == -1:
                continue
            relative_path = raw_path.replace("\\", "/")[idx:]
            wav_path = os.path.join(data_dir, *relative_path.split("/"))

            if os.path.exists(wav_path):
                files_to_process.append((wav_path, emotion))

        if files_to_process:
            return files_to_process

    # ---- Official release layout (Session*/dialog/{wav,label}) ----
    iemocap_root = data_dir
    if os.path.exists(os.path.join(data_dir, "IEMOCAP_full_unsegmentated")):
        iemocap_root = os.path.join(data_dir, "IEMOCAP_full_unsegmentated")

    emotion_map = {
        "ang": "angry",
        "hap": "happy",
        "sad": "sad",
        "neu": "neutral",
        "exc": "happy",
        "fru": "angry",
        "fea": "fearful",
        "dis": "disgust",
        "sur": "surprised",
        "xxx": None,
        "oth": None,
    }

    for subdir, _, files in os.walk(iemocap_root):
        for file in sorted(files):
            if file.endswith(".wav"):
                stem = os.path.splitext(file)[0]

                # Look for annotation in label folder
                possible_annot = None
                for label_dir in [
                    os.path.join(os.path.dirname(subdir), "label"),
                    os.path.join(iemocap_root, "label"),
                ]:
                    if os.path.exists(os.path.join(label_dir, f"{stem}.txt")):
                        possible_annot = os.path.join(label_dir, f"{stem}.txt")
                        break

                if possible_annot and os.path.exists(possible_annot):
                    with open(possible_annot, "r", encoding="utf-8") as f:
                        for line in f:
                            for code, emo in emotion_map.items():
                                if f"[{code}]" in line.upper() or f" {code} " in line.lower():
                                    if emo:
                                        files_to_process.append((os.path.join(subdir, file), emo))
                                    break

    return files_to_process


def load_emodb_dataset(data_dir):
    """Load EMODB dataset."""
    print("\nLoading EMODB dataset...")
    files_to_process = []

    # Try to load from wav2emotion.txt file
    wav2emotion_path = os.path.join(data_dir, "wav2emotion.txt")
    if os.path.exists(wav2emotion_path):
        with open(wav2emotion_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    wav_name = parts[0]
                    emo_code = parts[1].lower()

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
                    emo_name = label_map.get(emo_code, emo_code)

                    wav_path = os.path.join(data_dir, "wav", wav_name)
                    if os.path.exists(wav_path):
                        files_to_process.append((wav_path, emo_name))
    else:
        # Fallback: parse filenames directly.
        # Standard EMODB format: [speaker(2)][text code(3)][emotion(1)][version(1)].wav
        # e.g. "03a01Fa.wav" -> speaker=03, text=a01, emotion=F, version=a
        # Emotion letter is the second-to-last character of the filename stem.
        wav_dir = os.path.join(data_dir, "wav")
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
                        files_to_process.append((os.path.join(wav_dir, file), emotion))

    return files_to_process


def extract_features_for_files(files_to_process):
    """Extract features for a list of (filepath, emotion) tuples."""
    dataset = []
    failed = 0

    for i, (wav_path, emotion) in enumerate(files_to_process):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(files_to_process)}")

        if emotion not in FINE_TO_IDX:
            failed += 1
            continue

        features = extract_enhanced_features(wav_path)
        if features is not None:
            fine_idx = FINE_TO_IDX[emotion]
            coarse_idx = FINE_TO_COARSE[emotion]
            dataset.append((features, fine_idx, coarse_idx))
        else:
            failed += 1

    print(f"Extracted {len(dataset)} samples, {failed} failed")
    return dataset


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_on_dataset(model, test_loader, device, dataset_name):
    """Evaluate model on a single dataset."""
    model.eval()

    correct_fine = 0
    correct_coarse = 0
    total = 0

    all_fine_preds = []
    all_fine_labels = []
    all_coarse_preds = []
    all_coarse_labels = []

    with torch.no_grad():
        for X, f, c in test_loader:
            X = X.to(device)
            f = f.to(device)
            c = c.to(device)

            _, _, coarse_logits, fine_logits = model(X)

            fine_preds = fine_logits.argmax(1)
            coarse_preds = coarse_logits.argmax(1)

            correct_fine += (fine_preds == f).sum().item()
            correct_coarse += (coarse_preds == c).sum().item()
            total += f.size(0)

            all_fine_preds.extend(fine_preds.cpu().numpy())
            all_fine_labels.extend(f.cpu().numpy())
            all_coarse_preds.extend(coarse_preds.cpu().numpy())
            all_coarse_labels.extend(c.cpu().numpy())

    fine_acc = correct_fine / total
    coarse_acc = correct_coarse / total

    return {
        "dataset": dataset_name,
        "fine_accuracy": fine_acc,
        "coarse_accuracy": coarse_acc,
        "total_samples": total,
        "fine_preds": all_fine_preds,
        "fine_labels": all_fine_labels,
        "coarse_preds": all_coarse_preds,
        "coarse_labels": all_coarse_labels,
    }


def evaluate_ensemble(models, test_loader, device, dataset_name, weights=None):
    """Evaluate ensemble of models on a single dataset."""
    for model in models:
        model.eval()

    if weights is None:
        weights = [1 / len(models)] * len(models)

    weights = torch.tensor(weights, device=device).view(-1, 1, 1)

    correct_fine = 0
    correct_coarse = 0
    total = 0

    all_fine_preds = []
    all_fine_labels = []
    all_coarse_preds = []
    all_coarse_labels = []

    with torch.no_grad():
        for X, f, c in test_loader:
            X = X.to(device)
            f = f.to(device)
            c = c.to(device)

            fine_probs = []
            coarse_probs = []

            for model in models:
                _, _, coarse_logits, fine_logits = model(X)
                fine_probs.append(F.softmax(fine_logits, dim=1))
                coarse_probs.append(F.softmax(coarse_logits, dim=1))

            fine_probs = (
                torch.stack(fine_probs) * weights
            ).sum(dim=0)
            coarse_probs = (
                torch.stack(coarse_probs) * weights
            ).sum(dim=0)

            fine_preds = fine_probs.argmax(1)
            coarse_preds = coarse_probs.argmax(1)

            correct_fine += (fine_preds == f).sum().item()
            correct_coarse += (coarse_preds == c).sum().item()
            total += f.size(0)

            all_fine_preds.extend(fine_preds.cpu().numpy())
            all_fine_labels.extend(f.cpu().numpy())
            all_coarse_preds.extend(coarse_preds.cpu().numpy())
            all_coarse_labels.extend(c.cpu().numpy())

    fine_acc = correct_fine / total
    coarse_acc = correct_coarse / total

    return {
        "dataset": dataset_name,
        "fine_accuracy": fine_acc,
        "coarse_accuracy": coarse_acc,
        "total_samples": total,
        "fine_preds": all_fine_preds,
        "fine_labels": all_fine_labels,
        "coarse_preds": all_coarse_preds,
        "coarse_labels": all_coarse_labels,
    }


def save_detailed_results(results, results_dir):
    """Save detailed evaluation results."""
    os.makedirs(results_dir, exist_ok=True)

    # Create summary dataframe
    summary_data = []
    for result in results:
        summary_data.append({
            "Dataset": result["dataset"],
            "Fine Accuracy": f"{result['fine_accuracy']:.4f}",
            "Coarse Accuracy": f"{result['coarse_accuracy']:.4f}",
            "Total Samples": result["total_samples"],
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, "accuracy_comparison.csv"), index=False)
    print("\n" + "="*60)
    print("ACCURACY COMPARISON")
    print("="*60)
    print(summary_df.to_string(index=False))

    # Save confusion matrices and classification reports
    fine_emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    coarse_emotion_labels = ["neutral/calm", "happy/surprise", "sad/fear", "angry/disgust"]

    for result in results:
        dataset_name = result["dataset"]
        results_subdir = os.path.join(results_dir, dataset_name)
        os.makedirs(results_subdir, exist_ok=True)

        # Fine emotion confusion matrix
        cm_fine = confusion_matrix(result["fine_labels"], result["fine_preds"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_fine, annot=True, fmt="d", cmap="Blues",
                   xticklabels=fine_emotion_labels, yticklabels=fine_emotion_labels)
        plt.title(f"Fine Emotion Confusion Matrix - {dataset_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(results_subdir, "confusion_matrix_fine.png"), dpi=150)
        plt.close()

        # Coarse emotion confusion matrix
        cm_coarse = confusion_matrix(result["coarse_labels"], result["coarse_preds"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_coarse, annot=True, fmt="d", cmap="Blues",
                   xticklabels=coarse_emotion_labels, yticklabels=coarse_emotion_labels)
        plt.title(f"Coarse Emotion Confusion Matrix - {dataset_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(results_subdir, "confusion_matrix_coarse.png"), dpi=150)
        plt.close()

        # Fine emotion classification report
        report_fine = classification_report(
            result["fine_labels"], result["fine_preds"],
            target_names=fine_emotion_labels, output_dict=True
        )
        df_fine = pd.DataFrame(report_fine).transpose()
        df_fine.to_csv(os.path.join(results_subdir, "classification_report_fine.csv"))

        # Coarse emotion classification report
        report_coarse = classification_report(
            result["coarse_labels"], result["coarse_preds"],
            target_names=coarse_emotion_labels, output_dict=True
        )
        df_coarse = pd.DataFrame(report_coarse).transpose()
        df_coarse.to_csv(os.path.join(results_subdir, "classification_report_coarse.csv"))


def create_comparison_visualizations(results, results_dir):
    """Create comparison visualizations across datasets."""
    os.makedirs(results_dir, exist_ok=True)

    # Extract accuracies
    datasets = [r["dataset"] for r in results]
    fine_accs = [r["fine_accuracy"] * 100 for r in results]
    coarse_accs = [r["coarse_accuracy"] * 100 for r in results]

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(datasets))
    width = 0.35

    ax1.bar(x - width/2, fine_accs, width, label="Fine Emotion", color="skyblue")
    ax1.bar(x + width/2, coarse_accs, width, label="Coarse Emotion", color="lightcoral")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Model Accuracy Comparison Across Datasets")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (fine, coarse) in enumerate(zip(fine_accs, coarse_accs)):
        ax1.text(i - width/2, fine + 1, f"{fine:.2f}%", ha="center", va="bottom", fontsize=9)
        ax1.text(i + width/2, coarse + 1, f"{coarse:.2f}%", ha="center", va="bottom", fontsize=9)

    # Summary statistics table
    ax2.axis("tight")
    ax2.axis("off")

    table_data = []
    table_data.append(["Dataset", "Fine Acc (%)", "Coarse Acc (%)", "Samples"])
    for r in results:
        table_data.append([
            r["dataset"],
            f"{r['fine_accuracy']*100:.2f}",
            f"{r['coarse_accuracy']*100:.2f}",
            str(r["total_samples"])
        ])

    table = ax2.table(cellText=table_data, cellLoc="center", loc="center",
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.getcwd()
    results_base_dir = os.path.join(base_dir, "results", "dataset_evaluation")
    speech_dir = os.path.join(base_dir, "speech")

    # Load trained models
    print("\n" + "="*60)
    print("LOADING TRAINED MODELS")
    print("="*60)

    models = []
    seeds = [42, 7, 123, 2026, 99]

    for seed in seeds:
        model_path = os.path.join(base_dir, "results", f"best_model_seed_{seed}.pt")
        if os.path.exists(model_path):
            model = ARCANetV2().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            models.append(model)
            print(f"Loaded model: seed_{seed}")
        else:
            print(f"Model not found: seed_{seed}")

    if not models:
        print("ERROR: No trained models found!")
        return

    print(f"Successfully loaded {len(models)} models")

    # Evaluate on each dataset
    all_results = []

    datasets_config = [
        ("RAVDESS", load_ravdess_dataset, os.path.join(speech_dir, "RAVDESS")),
        ("IEMOCAP", load_iemocap_dataset, os.path.join(speech_dir, "IEMOCAP")),
        ("EMODB", load_emodb_dataset, os.path.join(speech_dir, "EMODB")),
    ]

    for dataset_name, loader_func, dataset_path in datasets_config:
        if not os.path.exists(dataset_path):
            print(f"\nWARNING: {dataset_name} dataset not found at {dataset_path}")
            print(f"  Please download and extract the {dataset_name} dataset")
            continue

        print(f"\n" + "="*60)
        print(f"EVALUATING ON {dataset_name}")
        print("="*60)

        # Load files
        files_to_process = loader_func(dataset_path)
        if not files_to_process:
            print(f"No files found for {dataset_name}")
            continue

        print(f"Found {len(files_to_process)} files")

        # Extract features
        dataset = extract_features_for_files(files_to_process)
        if not dataset:
            print(f"No valid samples extracted for {dataset_name}")
            continue

        # Create data loaders
        test_ds = SERDataset(dataset)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        # Evaluate ensemble
        result = evaluate_ensemble(models, test_loader, device, dataset_name)
        all_results.append(result)

        print(f"\n{dataset_name} Results:")
        print(f"  Fine Accuracy:   {result['fine_accuracy']:.4f}")
        print(f"  Coarse Accuracy: {result['coarse_accuracy']:.4f}")
        print(f"  Total Samples:   {result['total_samples']}")

    # Save and visualize results
    if all_results:
        print(f"\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)

        save_detailed_results(all_results, results_base_dir)
        create_comparison_visualizations(all_results, results_base_dir)

        # Save results as JSON
        json_results = []
        for result in all_results:
            json_results.append({
                "dataset": result["dataset"],
                "fine_accuracy": float(result["fine_accuracy"]),
                "coarse_accuracy": float(result["coarse_accuracy"]),
                "total_samples": int(result["total_samples"]),
            })

        with open(os.path.join(results_base_dir, "results.json"), "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to {results_base_dir}")


if __name__ == "__main__":
    main()
