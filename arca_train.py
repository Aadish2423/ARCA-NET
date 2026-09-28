import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from utils import get_dataloaders, HierarchicalMarginLoss
from arca_net import ARCANetV2


# ============================================================
# SEED CONTROL
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# CLASS WEIGHTS
# ============================================================

def class_weights(dataset, label_index, num_classes, device):

    labels = torch.tensor(
        [sample[label_index] for sample in dataset.data]
    )

    counts = torch.bincount(
        labels,
        minlength=num_classes
    ).float()

    weights = counts.sum() / (
        num_classes * counts.clamp(min=1)
    )

    return weights.to(device)


def class_counts(dataset, label_index, num_classes):

    labels = torch.tensor(
        [sample[label_index] for sample in dataset.data]
    )

    return torch.bincount(
        labels,
        minlength=num_classes
    ).float()


# ============================================================
# FEATURE AUGMENTATION
# ============================================================

def augment_features(x, noise_std=0.02, drop_prob=0.03):

    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    if drop_prob > 0:
        keep = torch.rand_like(x).ge(drop_prob).float()
        x = x * keep / (1.0 - drop_prob)

    return x


# ============================================================
# SAVE CLASSIFICATION OUTPUTS
# ============================================================

def save_classification_outputs(labels, preds, results_dir):

    os.makedirs(results_dir, exist_ok=True)

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    report = classification_report(labels, preds, output_dict=True)

    df = pd.DataFrame(report).transpose()

    df.to_csv(
        os.path.join(results_dir, "classification_report.csv")
    )

    print("Saved confusion matrix + classification report")


# ============================================================
# ENSEMBLE WEIGHT SEARCH
# ============================================================

def optimize_ensemble_weights(models, loader, device):

    if len(models) == 1:
        return [1.0]

    for model in models:
        model.eval()

    probs_all = []
    labels_all = []

    with torch.no_grad():

        for X, f, _ in loader:

            X = X.to(device)

            member_probs = []

            for model in models:

                _, _, _, logits = model(X)

                member_probs.append(
                    F.softmax(logits, dim=1)
                )

            probs_all.append(
                torch.stack(member_probs).cpu()
            )

            labels_all.append(f.cpu())

    probs_all = torch.cat(probs_all, dim=1)
    labels_all = torch.cat(labels_all)

    M = probs_all.size(0)

    best_weights = torch.ones(M) / M
    best_acc = 0

    for _ in range(256):

        w = torch.distributions.Dirichlet(
            torch.ones(M)
        ).sample()

        preds = (
            probs_all * w.view(-1, 1, 1)
        ).sum(dim=0).argmax(dim=1)

        acc = (preds == labels_all).float().mean()

        if acc > best_acc:

            best_acc = acc
            best_weights = w

    return best_weights.tolist()


# ============================================================
# EVALUATION
# ============================================================

def evaluate(models, loader, device, weights=None, return_preds=False):

    for model in models:
        model.eval()

    if weights is None:
        weights = [1 / len(models)] * len(models)

    weights = torch.tensor(
        weights,
        device=device
    ).view(-1, 1, 1)

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for X, f, _ in loader:

            X = X.to(device)
            f = f.to(device)

            probs = []

            for model in models:

                _, _, _, logits = model(X)

                probs.append(
                    F.softmax(logits, dim=1)
                )

            probs = (
                torch.stack(probs) * weights
            ).sum(dim=0)

            preds = probs.argmax(1)

            correct += (preds == f).sum().item()
            total += f.size(0)

            if return_preds:

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(f.cpu().numpy())

    acc = correct / total

    if return_preds:
        return acc, np.array(all_labels), np.array(all_preds)

    return acc


# ============================================================
# TRAIN LOOP
# ============================================================

def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    base_dir = os.getcwd()

    data_dir = os.path.join(base_dir, "speech")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    EPOCHS = 90
    BATCH_SIZE = 64
    LR = 8e-4

    SEEDS = [42, 7, 123, 2026, 99]

    train_loader, test_loader, total = get_dataloaders(
        data_dir,
        batch_size=BATCH_SIZE,
        balanced_sampling=True
    )

    print("Samples:", total)

    fine_weight = class_weights(
        train_loader.dataset, 1, 8, device
    )

    coarse_weight = class_weights(
        train_loader.dataset, 2, 4, device
    )

    fine_counts = class_counts(
        train_loader.dataset, 1, 8
    )

    coarse_counts = class_counts(
        train_loader.dataset, 2, 4
    )

    criterion = HierarchicalMarginLoss(
        coarse_weight=0.3,
        label_smoothing=0.08,
        fine_weight=fine_weight,
        coarse_class_weight=coarse_weight,
        fine_counts=fine_counts,
        coarse_counts=coarse_counts,
        capsule_weight=0.40
    )

    ensemble = []

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    for seed in SEEDS:

        print("\nSeed:", seed)

        set_seed(seed)

        model = ARCANetV2().to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=3e-3
        )

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LR,
            steps_per_epoch=len(train_loader),
            epochs=EPOCHS
        )

        best_acc = 0
        best_state = None

        epoch_losses = []
        epoch_accs = []

        for epoch in range(EPOCHS):

            model.train()

            running_loss = 0

            for X, f, c in train_loader:

                X, f, c = (
                    X.to(device),
                    f.to(device),
                    c.to(device)
                )

                X = augment_features(X)

                optimizer.zero_grad()

                _, _, cp, fp = model(X)

                loss = criterion(cp, fp, c, f)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            val_acc = evaluate(
                [model],
                test_loader,
                device
            )

            epoch_losses.append(avg_loss)
            epoch_accs.append(val_acc)

            if val_acc > best_acc:

                best_acc = val_acc
                best_state = copy.deepcopy(
                    model.state_dict()
                )

            if epoch % 5 == 0:

                print(
                    f"Epoch {epoch+1} | "
                    f"Loss {avg_loss:.4f} | "
                    f"Val Acc {val_acc:.4f}"
                )

        model.load_state_dict(best_state)

        torch.save(
            best_state,
            os.path.join(
                results_dir,
                f"best_model_seed_{seed}.pt"
            )
        )

        curve_df = pd.DataFrame({
            "loss": epoch_losses,
            "val_acc": epoch_accs
        })

        curve_df.to_csv(
            os.path.join(
                results_dir,
                f"training_curve_seed_{seed}.csv"
            ),
            index=False
        )

        ensemble.append(model)

        print(f"Seed {seed} training complete")


    # ========================================================
    # FINAL ENSEMBLE EVALUATION (SAFE)
    # ========================================================

    final_weights = optimize_ensemble_weights(
        ensemble,
        test_loader,
        device
    )

    final_acc, labels, preds = evaluate(
        ensemble,
        test_loader,
        device,
        final_weights,
        return_preds=True
    )

    save_classification_outputs(
        labels,
        preds,
        results_dir
    )

    with open(
        os.path.join(results_dir, "ensemble_weights.json"),
        "w"
    ) as f:

        json.dump(final_weights, f, indent=4)

    print("\nBest Ensemble Accuracy:", final_acc)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()