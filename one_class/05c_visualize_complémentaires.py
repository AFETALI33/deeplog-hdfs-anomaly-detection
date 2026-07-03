"""
DeepLog - Visualisations complémentaires FINALES (version corrigée)
====================================================================
CORRECTION : la courbe ROC utilise maintenant le split propre
(10 647 anomalies) depuis les fichiers .npy de 01_data_preparation.py
et non les 16 838 anomalies totales du CSV.
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({
    "font.family"       : "serif",
    "font.size"         : 11,
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 11,
    "legend.fontsize"   : 9,
    "figure.dpi"        : 150,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

# ─── CHEMINS ──────────────────────────────────────────────────
CSV_PATH    = "/content/drive/MyDrive/data_csv/"
DATA_PATH   = "/content/drive/MyDrive/data_finale_v3/"
RESULTS_DIR = "/content/drive/MyDrive/DeepLog3.0/results/"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures/")
VOCAB_SIZE  = 29
SEED        = 42


# ─── UTILITAIRES ──────────────────────────────────────────────

def comptage_session(seq, vocab_size=29):
    vec = np.zeros(vocab_size, dtype=np.float32)
    for e in seq:
        if 1 <= e <= vocab_size:
            vec[e - 1] += 1
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.logical_and(y_pred == 1, y_true == 1).sum()
    fp = np.logical_and(y_pred == 1, y_true == 0).sum()
    fn = np.logical_and(y_pred == 0, y_true == 1).sum()
    tn = np.logical_and(y_pred == 0, y_true == 0).sum()
    prec   = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1     = 2 * prec * recall / (prec + recall + 1e-10)
    fpr    = fp / (fp + tn + 1e-10)
    return {"precision": float(prec), "recall": float(recall),
            "f1": float(f1), "fpr": float(fpr),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn)}


def load_all_sessions():
    """
    Charge TOUTES les sessions (train + test) depuis le CSV.
    Utilisé uniquement pour les profils fréquentiels.
    """
    df  = pd.read_csv(os.path.join(CSV_PATH, "hdfs_sequences.csv"))
    tpl = pd.read_csv(os.path.join(CSV_PATH, "hdfs_templates.csv"))
    event_ids = sorted(tpl["EventId"].tolist())
    mapping   = {eid: idx + 1 for idx, eid in enumerate(event_ids)}

    sequences, labels, block_ids = [], [], []
    for _, row in df.iterrows():
        try:
            raw = ast.literal_eval(str(row["sequence"]))
            seq = [mapping[e] for e in raw]
            sequences.append(seq)
            labels.append(int(row["label"]))
            block_ids.append(row["block_id"])
        except Exception:
            continue
    return sequences, labels, block_ids


def split_sessions(sequences, labels, block_ids, train_ratio=0.8):
    """
    Reproduit exactement le split de 01_data_preparation.py.
    Utilisé uniquement pour les profils fréquentiels.
    """
    labels_arr  = np.array(labels)
    normal_idx  = np.where(labels_arr == 0)[0]
    anomaly_idx = np.where(labels_arr == 1)[0]

    rng = np.random.default_rng(42)
    rng.shuffle(normal_idx)

    n_train       = int(len(normal_idx) * train_ratio)
    train_idx     = normal_idx[:n_train]
    test_norm_idx = normal_idx[n_train:]
    test_idx      = np.concatenate([test_norm_idx, anomaly_idx])

    def get(idx_list):
        return ([sequences[i] for i in idx_list],
                [labels[i]    for i in idx_list],
                [block_ids[i] for i in idx_list])

    tr_s, tr_l, _       = get(train_idx)
    te_s, te_l, te_bids = get(test_idx)
    return tr_s, tr_l, te_s, te_l, te_bids


def session_level_from_windows(in_topk, block_ids_arr, y_ano_arr, k):
    window_pred  = (~in_topk[:, k - 1]).astype(np.int32)
    sort_idx     = np.argsort(block_ids_arr, kind="stable")
    bid_sorted   = block_ids_arr[sort_idx]
    pred_sorted  = window_pred[sort_idx]
    ano_sorted   = y_ano_arr[sort_idx]
    _, first_occ = np.unique(bid_sorted, return_index=True)
    s_pred = np.maximum.reduceat(pred_sorted, first_occ).clip(0, 1)
    s_true = np.maximum.reduceat(ano_sorted,  first_occ).clip(0, 1)
    return s_pred, s_true


# ─── FIGURE 1 : COURBE ROC + AUC ─────────────────────────────
# CORRECTION PRINCIPALE :
# On reconstruit les scores MLP directement sur les sessions
# du split propre (celles dont les block_ids sont dans
# block_ids_test.npy), pas sur toutes les anomalies du CSV.

def plot_roc_mlp_corrected(train_seqs, save_path):
    """
    Courbe ROC calculée sur le split propre de 01_data_preparation.py.

    MÉTHODE :
    1. On charge block_ids_test.npy et y_ano_test.npy
       (le split propre avec 10 647 anomalies)
    2. On recharge toutes les sessions du CSV
    3. On reconstruit le vecteur fréquentiel uniquement pour
       les sessions qui sont dans le split test propre
    4. On calcule le score de distance au centroïde
    5. On trace la ROC sur ces scores
    """
    print("  [ROC] Chargement du split propre depuis .npy...")

    # ── Charger le split propre ───────────────────────────────
    block_ids_test_arr = np.load(
        os.path.join(DATA_PATH, "block_ids_test.npy"))
    y_ano_test_arr = np.load(
        os.path.join(DATA_PATH, "y_ano_test.npy"))

    # ── Centroïde calculé sur le train ───────────────────────
    X_train = np.stack([comptage_session(s) for s in train_seqs])
    centroid = X_train.mean(axis=0)
    std      = X_train.std(axis=0) + 1e-8
    print(f"  [ROC] Centroïde calculé sur {len(X_train):,} "
          f"sessions normales train")

    # ── Charger toutes les sessions CSV ──────────────────────
    print("  [ROC] Chargement sessions CSV pour alignement...")
    df  = pd.read_csv(os.path.join(CSV_PATH, "hdfs_sequences.csv"))
    tpl = pd.read_csv(os.path.join(CSV_PATH, "hdfs_templates.csv"))
    event_ids = sorted(tpl["EventId"].tolist())
    mapping   = {eid: idx + 1 for idx, eid in enumerate(event_ids)}

    # Mapping block_id_string → block_id_int
    # (même ordre que dans 01_data_preparation.py)
    unique_blocks_csv = df["block_id"].unique()
    block_str_to_int  = {b: i for i, b in enumerate(unique_blocks_csv)}

    # Mapping block_id_int → séquence encodée
    block_int_to_seq = {}
    for _, row in df.iterrows():
        try:
            bid_str = row["block_id"]
            bid_int = block_str_to_int[bid_str]
            raw     = ast.literal_eval(str(row["sequence"]))
            seq     = [mapping[e] for e in raw]
            block_int_to_seq[bid_int] = seq
        except Exception:
            continue

    print(f"  [ROC] {len(block_int_to_seq):,} sessions mappées")

    # ── Construire scores sur le split propre ─────────────────
    # Pour chaque fenêtre du split test, on récupère la session
    # correspondante via son block_id_int

    # Les block_ids_test_arr contiennent des entiers répétés
    # (un par fenêtre). On veut UN score par session.
    sort_idx     = np.argsort(block_ids_test_arr, kind="stable")
    bid_sorted   = block_ids_test_arr[sort_idx]
    ano_sorted   = y_ano_test_arr[sort_idx]
    unique_bids, first_occ = np.unique(bid_sorted, return_index=True)

    # Label par session (OR logique sur les fenêtres)
    session_true = np.maximum.reduceat(
        ano_sorted, first_occ).clip(0, 1)

    # Score MLP par session
    scores_session = np.zeros(len(unique_bids), dtype=np.float32)
    n_found = 0
    for i, bid_int in enumerate(unique_bids):
        if bid_int in block_int_to_seq:
            seq   = block_int_to_seq[bid_int]
            vec   = comptage_session(seq)
            score = np.sqrt(((vec - centroid) / std) ** 2).sum()
            scores_session[i] = score
            n_found += 1

    print(f"  [ROC] Scores calculés pour {n_found:,} / "
          f"{len(unique_bids):,} sessions du split test")
    print(f"  [ROC] Dont {session_true.sum():,} sessions anormales")

    # ── Tracer la ROC ─────────────────────────────────────────
    fpr_arr, tpr_arr, thresholds_roc = roc_curve(
        session_true, scores_session)
    roc_auc = auc(fpr_arr, tpr_arr)

    optimal_idx = np.argmax(tpr_arr - fpr_arr)
    optimal_fpr = fpr_arr[optimal_idx]
    optimal_tpr = tpr_arr[optimal_idx]
    optimal_tau = thresholds_roc[optimal_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, color="#2563EB", lw=2,
            label=f"MLP one-class (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5,
            ls="--", label="Aléatoire (AUC = 0.50)")
    ax.fill_between(fpr_arr, tpr_arr, alpha=0.08,
                    color="#2563EB")
    ax.scatter(optimal_fpr, optimal_tpr,
               color="#DC2626", s=120, zorder=5, marker="*",
               label=f"Point optimal (τ={optimal_tau:.3f})\n"
                     f"TPR={optimal_tpr:.3f}  "
                     f"FPR={optimal_fpr:.3f}")
    ax.set_xlabel("Taux Faux Positifs (FPR)")
    ax.set_ylabel("Taux Vrais Positifs (TPR = Recall)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Courbe ROC — MLP One-class (centroïde fréquentiel)\n"
        f"Évaluée sur le split test propre "
        f"({int(session_true.sum()):,} sessions anormales)\n"
        f"AUC-ROC = {roc_auc:.4f}", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ ROC corrigée sauvegardée — AUC={roc_auc:.4f}")
    return roc_auc


# ─── FIGURE 2 : PROFILS FRÉQUENTIELS ─────────────────────────

def plot_frequency_profiles(train_seqs, test_seqs,
                             test_lbls, save_path):
    test_lbls_arr = np.array(test_lbls)
    X_train = np.stack([comptage_session(s) for s in train_seqs])
    X_test  = np.stack([comptage_session(s) for s in test_seqs])

    normal_mean  = X_train.mean(axis=0)
    anomaly_mask = test_lbls_arr == 1
    anomaly_mean = X_test[anomaly_mask].mean(axis=0)
    normal_std   = X_train.std(axis=0)

    events = [f"E{i+1}" for i in range(VOCAB_SIZE)]
    x      = np.arange(VOCAB_SIZE)
    w      = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.bar(x - w/2, normal_mean,  w, label="Normal (train)",
           color="#2563EB", alpha=0.8)
    ax.bar(x + w/2, anomaly_mean, w, label="Anomalie (test)",
           color="#DC2626", alpha=0.8)
    ax.fill_between(x - w/2,
                    normal_mean - normal_std,
                    normal_mean + normal_std,
                    alpha=0.15, color="#2563EB",
                    label="±1 std (normaux)")
    ax.set_xticks(x[::3])
    ax.set_xticklabels(events[::3], fontsize=8, rotation=30)
    ax.set_ylabel("Fréquence moyenne normalisée")
    ax.set_title("Profil fréquentiel moyen\nNormal vs Anomalie")
    ax.legend(fontsize=8)

    ax = axes[1]
    diff = np.abs(anomaly_mean - normal_mean)
    colors_bar = ["#DC2626" if d > np.percentile(diff, 75)
                  else "#6B7280" for d in diff]
    ax.bar(x, diff, color=colors_bar, alpha=0.85)
    ax.set_xticks(x[::3])
    ax.set_xticklabels(events[::3], fontsize=8, rotation=30)
    ax.set_ylabel("|μ_anomalie − μ_normale|")
    ax.set_title(
        "Différence absolue des profils\n"
        "(rouge = écart > 75e percentile)")

    most_diff = int(np.argmax(diff))
    ax.annotate(
        f"{events[most_diff]}\n+{diff[most_diff]:.3f}",
        xy=(most_diff, diff[most_diff]),
        xytext=(most_diff + 2, diff[most_diff] + 0.01),
        arrowprops=dict(arrowstyle="->", color="#374151"),
        fontsize=8)

    fig.suptitle(
        "Justification du détecteur MLP one-class — "
        "Les anomalies ont un profil fréquentiel distinct",
        fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Profils fréquentiels sauvegardés")


# ─── FIGURE 3 : SENSIBILITÉ SEUIL P80-P99 ────────────────────

def plot_threshold_sensitivity(train_seqs, test_seqs,
                                test_lbls, save_path):
    X_train = np.stack([comptage_session(s) for s in train_seqs])
    X_test  = np.stack([comptage_session(s) for s in test_seqs])
    y_true  = np.array(test_lbls)

    centroid     = X_train.mean(axis=0)
    std          = X_train.std(axis=0) + 1e-8
    scores_train = np.sqrt(
        ((X_train - centroid) / std) ** 2).sum(axis=1)
    scores_test  = np.sqrt(
        ((X_test  - centroid) / std) ** 2).sum(axis=1)

    percentiles  = list(range(80, 100))
    precisions, recalls, f1s, fprs, taus = [], [], [], [], []

    for p in percentiles:
        tau   = np.percentile(scores_train, p)
        preds = (scores_test > tau).astype(int)
        m     = compute_metrics(y_true, preds)
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])
        fprs.append(m["fpr"])
        taus.append(tau)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(percentiles, precisions, "o-", color="#2563EB",
            label="Précision", lw=2, markersize=4)
    ax.plot(percentiles, recalls,    "s-", color="#EA580C",
            label="Rappel",    lw=2, markersize=4)
    ax.plot(percentiles, f1s,        "^-", color="#16A34A",
            label="F1",        lw=2, markersize=4)
    ax.plot(percentiles, fprs,       "D-", color="#9333EA",
            label="FPR",       lw=1.5, markersize=4, alpha=0.7)
    ax.axvline(95, color="#374151", ls="--", lw=1.5,
               label="P95 choisi")
    ax.fill_between([93, 97], 0, 1, alpha=0.07,
                    color="#374151", label="Zone ±2%")
    ax.set_xlabel("Percentile du seuil τ")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Robustesse MLP — P/R/F1/FPR vs percentile\n"
        "(seuil calibré sur données train uniquement)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axis("off")
    markers_p = [90, 95, 99]
    rows = []
    for p in markers_p:
        idx = percentiles.index(p)
        rows.append([
            f"P{p}",
            f"τ = {taus[idx]:.3f}",
            f"{precisions[idx]:.4f}",
            f"{recalls[idx]:.4f}",
            f"{f1s[idx]:.4f}",
            f"{fprs[idx]:.4f}",
        ])
    table = ax.table(
        cellText=rows,
        colLabels=["Seuil", "τ",
                   "Précision", "Rappel", "F1", "FPR"],
        loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    for col in range(6):
        table[(2, col)].set_facecolor("#DBEAFE")
    ax.set_title(
        "Synthèse P90 / P95 / P99\n"
        "(P95 surligné = choix retenu)", pad=20)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Sensibilité seuil sauvegardée")


# ─── FIGURE 4 : MATRICE DE CONFUSION ─────────────────────────

def plot_confusion_matrix_combined(in_topk, block_ids_arr,
                                    y_ano_arr, k,
                                    train_seqs, save_path):
    """
    Deux matrices de confusion côte à côte :
    - Gauche  : LSTM seul K=5
    - Droite  : Combinaison LSTM(K=5) OR MLP one-class
    """

    # ── LSTM seul ─────────────────────────────────────────────
    s_pred_lstm, s_true = session_level_from_windows(
        in_topk, block_ids_arr, y_ano_arr, k)
    m_lstm = compute_metrics(s_true, s_pred_lstm)

    # ── MLP one-class sur le split propre ─────────────────────
    # Centroïde
    X_train  = np.stack([comptage_session(s) for s in train_seqs])
    centroid = X_train.mean(axis=0)
    std      = X_train.std(axis=0) + 1e-8
    tau      = np.percentile(
        np.sqrt(((X_train - centroid) / std) ** 2).sum(axis=1),
        95)

    # Reconstruire les scores MLP par session du split propre
    df  = pd.read_csv(os.path.join(CSV_PATH, "hdfs_sequences.csv"))
    tpl = pd.read_csv(os.path.join(CSV_PATH, "hdfs_templates.csv"))
    event_ids = sorted(tpl["EventId"].tolist())
    mapping   = {eid: idx + 1 for idx, eid in enumerate(event_ids)}

    unique_blocks_csv = df["block_id"].unique()
    block_str_to_int  = {b: i for i, b in
                         enumerate(unique_blocks_csv)}

    block_int_to_seq = {}
    for _, row in df.iterrows():
        try:
            bid_int = block_str_to_int[row["block_id"]]
            raw     = ast.literal_eval(str(row["sequence"]))
            block_int_to_seq[bid_int] = [mapping[e] for e in raw]
        except Exception:
            continue

    # Sessions uniques du split propre
    sort_idx   = np.argsort(block_ids_arr, kind="stable")
    bid_sorted = block_ids_arr[sort_idx]
    pred_lstm_sorted = s_pred_lstm[
        np.argsort(np.argsort(block_ids_arr, kind="stable"))]

    unique_bids, first_occ = np.unique(
        block_ids_arr, return_index=True)

    # Score MLP par session
    mlp_pred_session = np.zeros(len(unique_bids), dtype=np.int32)
    for i, bid_int in enumerate(unique_bids):
        if bid_int in block_int_to_seq:
            seq   = block_int_to_seq[bid_int]
            vec   = comptage_session(seq)
            score = np.sqrt(((vec - centroid) / std) ** 2).sum()
            mlp_pred_session[i] = int(score > tau)

    # Récupérer s_pred_lstm dans le même ordre que unique_bids
    sort_idx2    = np.argsort(block_ids_arr, kind="stable")
    bid_s2       = block_ids_arr[sort_idx2]
    pred_s2      = s_pred_lstm[
        np.searchsorted(
            np.unique(block_ids_arr),
            np.unique(block_ids_arr))]

    # Recalculer s_pred_lstm aligné sur unique_bids
    window_pred  = (~in_topk[:, k - 1]).astype(np.int32)
    sort_idx3    = np.argsort(block_ids_arr, kind="stable")
    bid_s3       = block_ids_arr[sort_idx3]
    pred_s3      = window_pred[sort_idx3]
    ano_s3       = y_ano_arr[sort_idx3]
    _, first3    = np.unique(bid_s3, return_index=True)
    lstm_by_bid  = np.maximum.reduceat(pred_s3, first3).clip(0,1)
    true_by_bid  = np.maximum.reduceat(ano_s3,  first3).clip(0,1)

    # Combinaison OR
    combined = np.logical_or(
        lstm_by_bid, mlp_pred_session).astype(int)
    m_comb   = compute_metrics(true_by_bid, combined)

    # ── Tracer les deux matrices ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, preds, m, title in [
        (axes[0], lstm_by_bid, m_lstm,
         f"LSTM seul (K={k})"),
        (axes[1], combined,    m_comb,
         f"LSTM(K={k}) OR MLP one-class"),
    ]:
        cm = np.array([
            [m["tn"], m["fp"]],
            [m["fn"], m["tp"]],
        ])
        total = cm.sum()
        im    = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(
            ["Prédit Normal", "Prédit Anomalie"], fontsize=10)
        ax.set_yticklabels(
            ["Réel Normal", "Réel Anomalie"], fontsize=10)
        ax.set_title(
            f"{title}\n"
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  FPR={m['fpr']:.4f}",
            fontsize=10)
        labels_cm = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val   = cm[i, j]
                pct   = 100 * val / total
                color = "white" if val > cm.max()/2 else "black"
                ax.text(j, i,
                        f"{labels_cm[i][j]}\n"
                        f"{val:,}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color=color)

    fig.suptitle(
        "Comparaison matrices de confusion — Session-level\n"
        "Gain du rappel par combinaison LSTM OR MLP",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Matrices combinées — "
          f"LSTM F1={m_lstm['f1']:.4f}  "
          f"Combiné F1={m_comb['f1']:.4f}")
    return m_lstm, m_comb


# ─── FIGURE 5 : DISTRIBUTION LONGUEUR FP/FN ──────────────────

def plot_session_length_analysis(in_topk, block_ids_arr,
                                  y_ano_arr, k, save_path):
    s_pred, s_true = session_level_from_windows(
        in_topk, block_ids_arr, y_ano_arr, k)

    sort_idx     = np.argsort(block_ids_arr, kind="stable")
    bid_sorted   = block_ids_arr[sort_idx]
    _, first_occ, counts = np.unique(
        bid_sorted, return_index=True, return_counts=True)

    fp_mask = np.logical_and(s_pred == 1, s_true == 0)
    fn_mask = np.logical_and(s_pred == 0, s_true == 1)
    tp_mask = np.logical_and(s_pred == 1, s_true == 1)
    tn_mask = np.logical_and(s_pred == 0, s_true == 0)

    fp_lengths = counts[fp_mask]
    fn_lengths = counts[fn_mask]
    tp_lengths = counts[tp_mask]
    tn_lengths = counts[tn_mask]

    def safe_mean(arr):
        return arr.mean() if len(arr) > 0 else 0.0

    print(f"\n  Analyse longueur sessions (K={k}) :")
    print(f"  TP : μ={safe_mean(tp_lengths):.1f}  "
          f"n={len(tp_lengths):,}")
    print(f"  TN : μ={safe_mean(tn_lengths):.1f}  "
          f"n={len(tn_lengths):,}")
    print(f"  FP : μ={safe_mean(fp_lengths):.1f}  "
          f"n={len(fp_lengths):,}")
    print(f"  FN : μ={safe_mean(fn_lengths):.1f}  "
          f"n={len(fn_lengths):,}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    bins = np.linspace(0, min(300, counts.max()), 50)
    if len(fp_lengths) > 0:
        ax.hist(fp_lengths, bins=bins, density=True, alpha=0.6,
                color="#DC2626",
                label=f"FP (n={len(fp_lengths):,}  "
                      f"μ={safe_mean(fp_lengths):.0f})")
    if len(fn_lengths) > 0:
        ax.hist(fn_lengths, bins=bins, density=True, alpha=0.6,
                color="#F59E0B",
                label=f"FN (n={len(fn_lengths):,}  "
                      f"μ={safe_mean(fn_lengths):.0f})")
    if len(tp_lengths) > 0:
        ax.hist(tp_lengths, bins=bins, density=True, alpha=0.3,
                color="#16A34A",
                label=f"TP (n={len(tp_lengths):,}  "
                      f"μ={safe_mean(tp_lengths):.0f})")
    ax.set_xlabel("Nombre de fenêtres par session")
    ax.set_ylabel("Densité")
    ax.set_title(
        f"Distribution longueur — FP, FN, TP  (K={k})")
    ax.legend(fontsize=8)

    ax = axes[1]
    categories  = ["TP", "TN", "FP", "FN"]
    means       = [safe_mean(tp_lengths), safe_mean(tn_lengths),
                   safe_mean(fp_lengths), safe_mean(fn_lengths)]
    counts_cat  = [len(tp_lengths), len(tn_lengths),
                   len(fp_lengths), len(fn_lengths)]
    bar_colors  = ["#16A34A", "#60A5FA", "#DC2626", "#F59E0B"]

    bars = ax.bar(categories, means,
                  color=bar_colors, alpha=0.85)
    for bar, mean, cnt in zip(bars, means, counts_cat):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + 1,
                f"μ={mean:.0f}\nn={cnt:,}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Longueur moyenne (nb fenêtres)")
    ax.set_title(
        "Longueur moyenne par catégorie\n"
        "(FN courts → peu de fenêtres pour détecter)")

    if safe_mean(fn_lengths) < safe_mean(tp_lengths):
        ax.text(0.5, 0.95,
                "→ Les anomalies manquées (FN) ont des\n"
                "   sessions plus courtes que les TP",
                transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
                color="#92400E",
                bbox=dict(boxstyle="round",
                          facecolor="#FEF3C7", alpha=0.8))

    fig.suptitle(
        f"Analyse des erreurs de classification — K={k}\n"
        f"Pourquoi le modèle se trompe ?",
        fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Analyse longueur sauvegardée")


# ─── MAIN ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DeepLog — Visualisations complémentaires FINALES")
    print("(version corrigée — ROC sur split propre)")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Chargement sessions pour profils fréquentiels ─────
    print("\n[INFO] Chargement sessions pour profils + seuil...")
    sequences, labels, block_ids = load_all_sessions()
    tr_s, tr_l, te_s, te_l, _ = split_sessions(
        sequences, labels, block_ids)

    print(f"  Train : {len(tr_s):,} sessions normales")
    print(f"  Test  (CSV split) : {len(te_s):,} sessions  "
          f"({sum(l==1 for l in te_l):,} anomalies)")

    # ── 2. Chargement résultats LSTM (split propre .npy) ─────
    print("\n[INFO] Chargement résultats LSTM (split propre)...")
    in_topk       = np.load(
        os.path.join(RESULTS_DIR, "in_topk.npy"))
    block_ids_arr = np.load(
        os.path.join(DATA_PATH, "block_ids_test.npy"))
    y_ano_arr     = np.load(
        os.path.join(DATA_PATH, "y_ano_test.npy"))

    # Nombre de sessions uniques dans le split propre
    n_sessions_proper = len(np.unique(block_ids_arr))
    n_ano_proper      = int(np.maximum.reduceat(
        y_ano_arr[np.argsort(block_ids_arr, kind="stable")],
        np.unique(block_ids_arr,
                  return_index=True)[1]).clip(0,1).sum())

    print(f"  Split propre : {n_sessions_proper:,} sessions  "
          f"({n_ano_proper:,} anomalies)")

    # ── 3. Génération des figures ─────────────────────────────
    print("\n📊 Génération des 5 figures...\n")

    # Figure 1 : ROC corrigée sur split propre
    print("[1/5] Courbe ROC MLP (split propre)...")
    roc_auc = plot_roc_mlp_corrected(
        tr_s,
        os.path.join(FIGURES_DIR, "fig_roc_mlp.pdf"))

    # Figure 2 : Profils fréquentiels
    print("[2/5] Profils fréquentiels Normal vs Anomalie...")
    plot_frequency_profiles(
        tr_s, te_s, te_l,
        os.path.join(FIGURES_DIR, "fig_freq_profiles.pdf"))

    # Figure 3 : Sensibilité seuil
    print("[3/5] Sensibilité seuil P80-P99...")
    plot_threshold_sensitivity(
        tr_s, te_s, te_l,
        os.path.join(FIGURES_DIR, "fig_threshold_sens.pdf"))

    # Figure 4 : Matrice de confusion K=5 (split propre)
    print("[4/5] Matrice de confusion K=5...")
    m_k5 = plot_confusion_matrix_combined(
        in_topk, block_ids_arr, y_ano_arr, k=5,
        save_path=os.path.join(
            FIGURES_DIR, "fig_confusion_matrix.pdf"))

    # Figure 5 : Distribution longueur FP/FN (split propre)
    print("[5/5] Distribution longueur sessions FP/FN...")
    plot_session_length_analysis(
        in_topk, block_ids_arr, y_ano_arr, k=5,
        save_path=os.path.join(
            FIGURES_DIR, "fig_dist_longueur.pdf"))

    # ── 4. Résumé ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RÉSUMÉ — toutes métriques sur split propre")
    print("=" * 60)
    print(f"  AUC-ROC MLP (split propre) : {roc_auc:.4f}")
    print(f"  Session K=5  :  "
          f"P={m_k5['precision']:.4f}  "
          f"R={m_k5['recall']:.4f}  "
          f"F1={m_k5['f1']:.4f}")
    print(f"\n  Figures dans : {FIGURES_DIR}")
    print(f"    fig_roc_mlp.pdf          ← ROC corrigée")
    print(f"    fig_freq_profiles.pdf")
    print(f"    fig_threshold_sens.pdf")
    print(f"    fig_confusion_matrix.pdf")
    print(f"    fig_dist_longueur.pdf")
    print("=" * 60)
    print("\n✅ Terminé — toutes métriques sur split propre.")


if __name__ == "__main__":
    main()