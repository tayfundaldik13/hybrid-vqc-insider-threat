PAPER_ARCH = {
    "r4.2":   dict(INPUT_DIM=16, N_QUBITS=9, N_LAYERS=2, PRE_HIDDEN=0,   NN_HIDDEN=8, NN_DEPTH=1),
    "r5.2":   dict(INPUT_DIM=16, N_QUBITS=9, N_LAYERS=4, PRE_HIDDEN=32,  NN_HIDDEN=8, NN_DEPTH=2),
    "spedia": dict(INPUT_DIM=16, N_QUBITS=7, N_LAYERS=2, PRE_HIDDEN=128, NN_HIDDEN=8, NN_DEPTH=2),
}
PAPER_N_PARAMS = {"r4.2": 8848, "r5.2": 17670, "spedia": 18836}


def check_config(cfg, dataset):
    want = PAPER_ARCH[dataset]
    got = {k: getattr(cfg, k, None) for k in want}
    if got != want:
        raise RuntimeError(f"Quantum config for {dataset} ({getattr(cfg, '__file__', cfg)}) is {got}, "
                           f"paper architecture is {want}")
    return want
