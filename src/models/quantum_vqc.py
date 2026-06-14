"""
Hybrid Quantum VQC Model for Insider Threat Detection
"""
import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
from pennylane.qnn import TorchLayer
import os


def create_vqc_circuit(n_qubits=8, n_layers=3):
    dev = qml.device("default.qubit", wires=n_qubits)
    diff_method = "backprop"
    print(f"VQC device: default.qubit ({n_qubits} qubits, backprop diff)")

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AmplitudeEmbedding(
            features=inputs,
            wires=range(n_qubits),
            normalize=True,
            pad_with=0.0,
        )
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}

    return circuit, weight_shapes


class HybridQuantumModel(nn.Module):
    def __init__(self, input_dim=512, n_qubits=8, n_layers=3,
                 nn_hidden=16, nn_depth=1, pre_hidden=0):
        super().__init__()
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self.amp_dim = 2 ** n_qubits  # AmplitudeEmbedding requires 2^n_qubits features

        if pre_hidden > 0:
            self.pre_quantum = nn.Sequential(
                nn.Linear(input_dim, pre_hidden),
                nn.ReLU(),
                nn.Linear(pre_hidden, self.amp_dim),
                nn.Tanh(),
            )
        else:
            self.pre_quantum = nn.Sequential(
                nn.Linear(input_dim, self.amp_dim),
                nn.Tanh(),
            )

        circuit, weight_shapes = create_vqc_circuit(n_qubits, n_layers)
        self.vqc = TorchLayer(circuit, weight_shapes)

        if nn_depth == 1:
            self.post_quantum = nn.Sequential(
                nn.Linear(n_qubits, nn_hidden),
                nn.ReLU(),
                nn.Linear(nn_hidden, 1),
            )
        elif nn_depth == 2:
            self.post_quantum = nn.Sequential(
                nn.Linear(n_qubits, nn_hidden),
                nn.ReLU(),
                nn.Linear(nn_hidden, nn_hidden // 2),
                nn.ReLU(),
                nn.Linear(nn_hidden // 2, 1),
            )
        else:
            self.post_quantum = nn.Linear(n_qubits, 1)

        # Temperature Scaling (Guo et al., 2017: "On Calibration of Modern Neural Networks")
        self.temperature = nn.Parameter(torch.tensor(1.5))

    def forward(self, x):
        x = self.pre_quantum(x)
        x_norm = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        vqc_out = self.vqc(x_norm)
        logits = self.post_quantum(vqc_out.float())
        return logits / self.temperature.clamp(min=0.5, max=10.0)

def save_hybrid_checkpoint(model, optimizer, epoch, fold, stage_size, seed, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"hybrid_seed{seed}_size{stage_size}_fold{fold}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "fold": fold,
        "stage_size": stage_size,
        "seed": seed,
    }, path)
    print(f"Hybrid checkpoint: {os.path.basename(path)}")
    return path


def load_hybrid_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return model, optimizer

    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    prev = ckpt.get("stage_size", "?")
    print(f"Loaded hybrid checkpoint: stage={prev}")
    return model, optimizer


def get_hybrid_checkpoint_path(checkpoint_dir, seed, stage_size, fold):
    return os.path.join(checkpoint_dir, f"hybrid_seed{seed}_size{stage_size}_fold{fold}.pt")
