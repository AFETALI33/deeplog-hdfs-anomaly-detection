# DeepLog — Détection d'anomalies dans les logs HDFS
# DeepLog — Anomaly Detection in HDFS System Logs

**Mémoire de fin d'études — Master 2 Réseaux, Sécurité et Systèmes Embarqués**  
**Master's Thesis — Networks, Security and Embedded Systems**

| | |
|---|---|
| **Établissement / Institution** | Centre de Recherche en Information Scientifique et Technique (CERIST), Alger |
| **Auteurs / Authors** | BOUBCHIR Abderrazek · BOUZELBOUDJEN Mohamed Abdelwahab |
| **Encadrants / Supervisors** | RAHMANI Amine · DERKI Mohamed Saddek |
| **Année / Year** | 2026 |
| **Note / Grade** | 15.5 / 20 |

---

## Résumé / Abstract

**FR** — Ce dépôt contient l'implémentation complète du pipeline de détection d'anomalies dans les logs systèmes HDFS, développé dans le cadre de notre mémoire de Master 2. Le système explore deux protocoles complémentaires : un protocole **one-class (non supervisé)** combinant DeepLog LSTM et MLP centroïde fréquentiel, et un protocole **semi-supervisé** exploitant 70% des anomalies disponibles à l'entraînement.

**EN** — This repository contains a full anomaly detection pipeline for HDFS system logs, built as part of our Master's thesis. Two complementary protocols are explored: a **one-class (unsupervised)** protocol combining DeepLog LSTM and a frequency-centroid MLP, and a **semi-supervised** protocol that uses 70% of available anomalies during training.

---

## Résultats principaux / Main Results

### Protocole one-class — One-class Protocol (test : 10 647 anomalies)

| Système / System | Précision / Precision | Rappel / Recall | F1 | FPR |
|---|---|---|---|---|
| LSTM seul (K=5) | 0.973 | 0.439 | 0.605 | 0.001 |
| MLP one-class (P99) | 0.948 | 1.000 | **0.973** | 0.008 |
| LSTM OR MLP | 0.707 | 1.000 | 0.829 | 0.039 |

### Protocole semi-supervisé — Semi-supervised Protocol (test : 5 052 anomalies)

| Système / System | Précision / Precision | Rappel / Recall | F1 | FPR |
|---|---|---|---|---|
| LSTM seul (K=3) | 0.769 | 0.655 | 0.707 | 0.009 |
| MLP supervisé | 0.9996 | 0.9984 | **0.9990** | ~0.00 |
| LSTM OR MLP | 0.835 | 0.9996 | 0.910 | 0.009 |

> **Note** : Les deux protocoles utilisent des ensembles de test différents (10 647 vs 5 052 anomalies). Une comparaison directe n'est pas statistiquement valide — voir section 5.4 du mémoire.  
> **Note**: Both protocols use different test sets (10,647 vs 5,052 anomalies). Direct comparison is not statistically valid — see thesis section 5.4.

**Contribution analytique clé / Key analytical finding** : quantification du mismatch fenêtre-level vs session-level. Pour K=5, le F1 passe de 0.099 à 0.605, soit un facteur ×6.1.

![Mismatch fenêtre vs session](results/figures/fig6_topk_window_vs_session.png)

---

## Architecture du modèle LSTM / LSTM Architecture

```
Entrée / Input : séquence de 10 EventIDs     [batch × 10]
    ↓
Embedding (30 → 64)                          [batch × 10 × 64]
    ↓
LSTM couche 1 (64 → 128)                    [batch × 10 × 128]
    ↓
Dropout (p=0.2)
    ↓
LSTM couche 2 (128 → 128)                   [batch × 10 × 128]
    ↓
Extraction dernier état [:, −1, :]          [batch × 128]
    ↓
Linéaire + Softmax (128 → 30)               [batch × 30]

Paramètres totaux / Total parameters : 237 214
Entraînement / Training : ~27 min — Tesla T4 — 26 epochs
```

---

## Dataset

Le dataset **HDFS v1** est disponible sur le dépôt officiel / available on the official repository :  
→ https://github.com/logpai/loghub/tree/master/HDFS

Fichiers nécessaires / Required files :
- `HDFS.log` — logs bruts / raw logs (~1.5 GB)
- `anomaly_label.csv` — labels par BlockID / labels per BlockID

Ces fichiers ne sont **pas inclus** dans ce dépôt. Téléchargez-les et placez-les dans `data/raw/` avant d'exécuter le pipeline.  
These files are **not included** in this repo. Download them and place them in `data/raw/` before running the pipeline.

---

## Structure du dépôt / Repository Structure

```
deeplog-hdfs-anomaly-detection/
│
├── 00_parse_hdfs_logs.py          # Parsing HDFS.log → séquences + templates
│                                  # (commun aux deux protocoles / shared by both)
│
├── one_class/                     # Protocole one-class (non supervisé)
│   ├── 01_data_preparation.py     # Fenêtrage, split train/test
│   ├── 02_model.py                # Architecture DeepLog LSTM
│   ├── 03_train.py                # Entraînement LSTM
│   ├── 04_evaluate.py             # Inférence Top-K + métriques session-level
│   ├── 05_visualize.py            # Figures principales (fig1–fig9)
│   ├── 05b_visualize_complement.py
│   ├── 05c_visualize_complémentaires.py  # ROC, profils fréquentiels
│   ├── 06_mlp_counting.py         # MLP one-class centroïde + combinaison
│   └── 07_robustness.py           # Robustesse + baselines triviales
│
├── semisupervised/                # Protocole semi-supervisé
│   ├── deeplog_combined.py        # LSTM + MLP supervisé (70% anomalies)
│   └── deeplogsemisup2.py
│
├── results/
│   └── figures/                   # 16 figures exportées (PDF + PNG)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/AFITALI33/deeplog-hdfs-anomaly-detection.git
cd deeplog-hdfs-anomaly-detection
pip install -r requirements.txt
```

---

## Ordre d'exécution / Execution Order

### Étape préalable / Prerequisites — Télécharger le dataset / Download the dataset

```
data/
└── raw/
    ├── HDFS.log
    └── anomaly_label.csv
```

### Étape 0 — Parsing (commun aux deux protocoles / shared)

```bash
python 00_parse_hdfs_logs.py
```

Produit dans `data/csv/` :
- `hdfs_sequences.csv` — séquences par BlockID + labels
- `hdfs_templates.csv` — 29 templates EventID

### Protocole one-class (`one_class/`)

```bash
python one_class/01_data_preparation.py   # Fenêtrage + split
python one_class/03_train.py              # Entraînement LSTM (GPU recommandé)
python one_class/04_evaluate.py           # Évaluation Top-K session-level
python one_class/05_visualize.py          # Figures principales
python one_class/06_mlp_counting.py       # MLP centroïde + combinaison LSTM OR MLP
python one_class/07_robustness.py         # Robustesse + baselines
python one_class/05c_visualize_complémentaires.py  # ROC, profils fréquentiels
```

### Protocole semi-supervisé (`semisupervised/`)

```bash
python semisupervised/deeplog_combined.py   # LSTM + MLP supervisé
```

---

## Hyperparamètres / Hyperparameters

| Paramètre | Valeur | Description |
|---|---|---|
| window_size | 10 | Taille fenêtre glissante |
| vocab_size | 30 | 29 EventIDs + 1 padding |
| embed_dim | 64 | Dimension embedding |
| hidden_size | 128 | Neurones par couche LSTM |
| num_layers | 2 | Couches LSTM empilées |
| dropout | 0.2 | Taux dropout inter-couches |
| batch_size | 2048 | Taille batch entraînement |
| lr_max | 3e-3 | LR max OneCycleLR |
| epochs | 30 | Epochs max (early stop p=5) |
| K_values | [1,3,5,7,9,11,15] | Valeurs Top-K évaluées |
| percentile_seuil | 95 / 99 | Calibration seuil MLP |

---

## Analyse Top-K — Mismatch fenêtre vs session

| K | F1 fenêtre | F1 session | Gain |
|---|---|---|---|
| 1 | 0.270 | 0.379 | +40% |
| 3 | 0.149 | 0.491 | +230% |
| 5 | 0.099 | 0.605 | **+511%** |
| 9 | 0.054 | 0.367 | +579% |

---

## Références / References

```bibtex
@inproceedings{du2017deeplog,
  author    = {Du, Min and Li, Feifei and Zheng, Guineng and Srikumar, Vivek},
  title     = {DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning},
  booktitle = {ACM CCS},
  year      = {2017},
  pages     = {1285--1298}
}

@inproceedings{he2016experience,
  author    = {He, Pinjia and Zhu, Jieming and He, Shilin and Li, Jian and Lyu, Michael R.},
  title     = {An Evaluation Study on Log Parsing and Its Use in Log Mining},
  booktitle = {DSN},
  year      = {2016}
}

@inproceedings{meng2019loganomaly,
  author    = {Meng, Weibin and others},
  title     = {LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs},
  booktitle = {IJCAI},
  year      = {2019}
}

@inproceedings{guo2021logbert,
  author    = {Guo, Haixuan and Yuan, Shuhan and Wu, Xintao},
  title     = {LogBERT: Log Anomaly Detection via BERT},
  booktitle = {IJCNN},
  year      = {2021}
}
```

---

## Licence / License

Code publié à des fins académiques — Mémoire Master 2, CERIST 2026.  
Toute réutilisation doit citer ce travail et les références associées.

Published for academic purposes — Master's thesis, CERIST 2026.  
Any reuse must cite this work and the associated references.
