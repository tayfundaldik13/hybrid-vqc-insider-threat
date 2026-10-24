# Reliable Insider Threat Detection Under Data Scarcity: A Hybrid Variational Quantum Classifier

Code and per-fold results for the paper by Fatih Şahin and Erdoğan Tayfun Daldık (IEEE Access, under review).

The Hybrid-VQC maps 16 PCA components of daily user behavior to a 9-qubit amplitude-encoded state, processes it with a StronglyEntanglingLayers ansatz, and classifies the Pauli-Z expectation values with a small classical head and learned temperature scaling. It is compared with nine classical baselines on CERT r4.2, CERT r5.2, and SPEDIA at training sizes of 50, 100, 250, and 500 samples, and on the full CERT r4.2 dataset. All quantum results come from noiseless state-vector simulation.

## Model

```
PCA(16) -> Linear(16 -> 2^n) + tanh -> L2 normalization -> AmplitudeEmbedding(n qubits)
        -> StronglyEntanglingLayers(L) -> <Z_i>, i = 1..n -> MLP head -> logit / T
```

| Dataset   | Qubits | Layers | Pre-quantum         | Head          | Trainable parameters |
|-----------|--------|--------|---------------------|---------------|----------------------|
| CERT r4.2 | 9      | 2      | 16 -> 512           | 9 -> 8 -> 1   | 8,848                |
| CERT r5.2 | 9      | 4      | 16 -> 32 -> 512     | 9 -> 8 -> 4 -> 1 | 17,670            |
| SPEDIA    | 7      | 2      | 16 -> 128 -> 128    | 7 -> 8 -> 4 -> 1 | 18,836            |

The temperature T is a learned parameter (initial value 1.5, clamped to [0.5, 10]). The full-data CERT r4.2 experiment uses the CERT r4.2 architecture.

## Baselines

Naive Bayes, linear SVM, BiLSTM (Singh et al.), Random Forest, XGBoost, MLP, Transformer encoder, ResHybNet, and SCARF (self-supervised). All models receive the same 16 PCA features and the same outer folds.

## Protocol

- 10 random seeds (42 to 51) x 5 outer folds = 50 paired observations per model, dataset, and training size.
- Inner Optuna search (TPE, 10 trials for the Hybrid-VQC) on a 75/25 split of each outer training fold; the Hybrid-VQC inner search tunes learning rate, batch size, and number of epochs. The architecture is fixed by the global search.
- Classical baseline hyperparameters are in the `config.py` files.
- Statistics: two-sided Wilcoxon signed-rank tests on the 50 paired values, Benjamini-Hochberg correction over all 324 tests, Cohen's d, mean ranks, and Shapiro-Wilk tests.

## Repository structure

```
src/                 CERT r4.2 configuration, data processing, models, metrics
main_*.py            CERT r4.2 progressive experiments, one script per model
parallel_quantum.py  parallel runner for the Hybrid-VQC (worker / aggregate)
run_parallel_quantum.sh, quantum_status.sh
optuna/              hyperparameter search scripts for CERT r4.2
ablation_quantum.py  ablation study on CERT r4.2
preprocess_behavioral.py  CERT r4.2 feature extraction
r5.2/                CERT r5.2 (same layout: src/, main_*.py, optuna/, preprocess)
spedia/              SPEDIA (same layout: src/, main_*.py, optuna/, preprocess_spedia.py)
all_data/            full-data CERT r4.2 experiment (main_*_all.py, parallel runner, status_parallel.sh)
stats/paper_statistics.py  reproduces the statistical tables from the result files
results/             per-fold results used in the paper
data/                preprocessed CERT r4.2 features (Git LFS); r5.2/data/ and spedia/data/ hold the other datasets
```

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3.10 on Linux.

## Running

Each script is run from its own dataset folder:

```
python main_quantum.py            # CERT r4.2, Hybrid-VQC
python main_rf.py                 # CERT r4.2, Random Forest (same pattern for every baseline)
cd r5.2 && python main_quantum.py
cd spedia && python main_quantum.py
cd all_data && python main_quantum_all.py
```

The Hybrid-VQC can also be run in parallel with `run_parallel_quantum.sh` (and `all_data/run_parallel_quantum_all.sh`); `quantum_status.sh` reports progress.

The raw CERT logs are not included. To rebuild the feature files, place the CERT r4.2 release and the answers folder under `raw_data/r4.2` and `raw_data/answers`, and the r5.2 release under `r5.2/raw_data/r5.2` and `r5.2/raw_data/answers` (or set `CERT_R42_DIR`, `CERT_R52_DIR`, `CERT_ANSWERS_DIR`), then run `preprocess_behavioral.py` in the corresponding folder.

Each dataset folder has an `optuna/` folder with one search script per model, run from that dataset folder (for example `python optuna/optuna_quantum.py` or `cd r5.2 && python optuna/optuna_quantum.py`). The scripts write the best values they find into the configuration files. The configuration files already contain the values used in the paper, so the search does not need to be rerun to reproduce the results.

## Statistics

```
python stats/paper_statistics.py --r42 results/cert_r4.2 --r52 results/cert_r5.2 --spedia results/spedia
```

This prints the Wilcoxon summary, the Cohen's d tables, the mean ranks, and the Shapiro-Wilk tests reported in the paper, and writes all 324 test results to `statistics_output/wilcoxon_all_tests.csv`.

## Citation

```
F. Şahin and E. T. Daldık, "Reliable Insider Threat Detection Under Data Scarcity: A Hybrid Variational Quantum Classifier," IEEE Access, under review.
```
