
# Hybrid Variational Quantum Circuit for Insider Threat Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/) [![PennyLane](https://img.shields.io/badge/PennyLane-0.36%2B-green)](https://pennylane.ai/) 

> **Hybrid Quantum-Classical Neural Network architecture combining Variational Quantum Circuits (VQC).**

---

## Overview

This repository contains the full reproducibility bundle for the paper:

**"A Hybrid Variational Quantum Circuit Approach for Insider Threat Detection"**  
_Erdoğan Tayfun Daldık, Dr. Fatih Şahin 
The core contribution is a **Hybrid-VQC** model that integrates:

- PennyLane `StronglyEntanglingLayers`
- Amplitude embedding of behavioral feature vectors
- Pauli-Z measurement → classical post-quantum head
- Temperature scaling for calibration
- Evaluated against BiLSTM (Singh et al.), SVM, and Naive Bayes baselines

A key empirical finding is **BiLSTM catastrophic degradation** on CERT r5.2 as training size grows, while Hybrid-VQC maintains stable, reliable performance — making it particularly suited for real-world insider threat scenarios where labeled data is scarce.

---

## Repository Structure

```
hybrid-vqc-insider-threat/
├── data/                          # CERT r4.2 preprocessed features
├── r5.2/                          # CERT r5.2 dataset pipeline
├── spedia/                        # SPEDIA dataset pipeline
├── optuna/                        # Hyperparameter search logs
├── src/                           # Shared utilities and modules
├── preprocess_behavioral.py       # Feature engineering — CERT r4.2 / r5.2
├── preprocess_spedia.py           # Feature engineering — SPEDIA
├── main_quantum.py                # Hybrid-VQC training & evaluation
├── main_nb.py                     # Naive Bayes baseline
├── main_svm.py                    # SVM baseline
├── main_singh.py                  # Singh et al. replication
├── ablation_quantum.py            # 12-variant ablation study for CERT r4.2
├── .gitattributes            
└── .gitignore
```

---

## Datasets

|Dataset|Version|Samples|Features|Threat Ratio|
|---|---|---|---|---|
|CERT Insider Threat|r4.2|~70K|26|~1%|
|CERT Insider Threat|r5.2|~70K|29|~1%|
|SPEDIA|—|variable|—|balanced|

> **Note:** Raw CERT dataset files are **not included** in this repository. Request access at [https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099).  
> The `.pkl` files (preprocessed behavioral features) are tracked via **Git LFS**.

---

## Architecture

```
Input Features (26/29-dim)
        │
   [Amplitude Embedding]
        │
   [StronglyEntanglingLayers]
   9 qubits × 2 layers
   Ring CNOT entanglement
        │
   [Pauli-Z Measurement]
        │
   [Classical Head]
   Dense → BN → Dropout → Dense
        │
   Binary Output (Insider / Normal)
```

---

## Installation

```bash
git clone https://github.com/tayfundaldik13/hybrid-vqc-insider-threat.git
cd hybrid-vqc-insider-threat

# Install dependencies
pip install pennylane torch scikit-learn numpy pandas optuna

# Pull LFS files (large .pkl files)
git lfs pull
```

### Requirements

- Python 3.9+
- PennyLane ≥ 0.36
- PyTorch ≥ 2.0
- scikit-learn
- Optuna (hyperparameter search)
- NumPy, Pandas

---

## Usage

### 1. Preprocess

```bash
# CERT r4.2 / r5.2
python preprocess_behavioral.py

# SPEDIA
python preprocess_spedia.py
```

### 2. Train Hybrid-VQC

```bash
python main_quantum.py
```

### 3. Run Baselines

```bash
python main_svm.py
python main_nb.py
python main_singh.py
```

### 4. Ablation Study (12 variants)

```bash
python ablation_quantum.py
```

---

## Results Summary

Hybrid-VQC consistently outperforms classical baselines in **MCC and Specificity** under low-data conditions across all three datasets. Full tables and figures are available in the paper.

|Model|Dataset|F1|MCC|Specificity|
|---|---|---|---|---|
|Hybrid-VQC|CERT r4.2|—|—|—|
|Hybrid-VQC|CERT r5.2|—|—|—|
|BiLSTM|CERT r5.2|↘ degrades|↘|↘|
|SVM|CERT r4.2|—|—|—|

_(Full results in paper — table cells to be filled post-acceptance)_

---

## Contact

**Erdoğan Tayfun Daldık**  
Computer Engineering, Topkapı University, Istanbul  
Supervised by Dr. Fatih Şahin (Topkapı University)  
GitHub: [@tayfundaldik13](https://github.com/tayfundaldik13)
