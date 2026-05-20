"""Utility functions for SmoothGNN experiments on .mat graph anomaly datasets."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import dgl
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from scipy.sparse import csgraph
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import degree


def set_seed(seed: int) -> int:
    """Set random seed. If seed <= 0, use current timestamp."""
    if seed <= 0:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def _first_key(mat: dict, candidates: tuple[str, ...]):
    lower_map = {k.lower(): k for k in mat.keys() if not k.startswith("__")}
    for name in candidates:
        if name in mat:
            return mat[name]
        key = lower_map.get(name.lower())
        if key is not None:
            return mat[key]
    raise KeyError(f"Cannot find any key from {candidates}. Available keys: {list(lower_map.values())}")


def _to_dense_float_tensor(x) -> torch.Tensor:
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return torch.from_numpy(x).float()


def _to_label_tensor(y) -> torch.Tensor:
    if sp.issparse(y):
        y = y.toarray()
    y = np.asarray(y).reshape(-1)
    # Some datasets use {-1, 1}; convert the larger/positive anomaly label to 1.
    uniq = np.unique(y)
    if set(uniq.tolist()) == {-1, 1}:
        y = (y == 1).astype(np.int64)
    else:
        y = (y > 0).astype(np.int64)
    return torch.from_numpy(y).long()


def _edge_index_from_adj(adj) -> torch.Tensor:
    if not sp.issparse(adj):
        adj = sp.coo_matrix(adj)
    else:
        adj = adj.tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64))
    col = torch.from_numpy(adj.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def load_data(dataset: str, data_dir: str | Path = "~/datasets/GAD/mat") -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a GAD .mat dataset and return DGL graph, features, labels, edge_index and eval index.

    Supported common keys:
      adjacency: Network / network / A / adj / Adj / adjacency
      features: Attributes / attributes / X / x / feature / features / attr
      labels: Label / label / y / gnd / anomaly_label

    If <dataset>_index.txt exists beside the .mat file, it is used as eval index;
    otherwise all labelled nodes are evaluated.
    """
    data_dir = Path(data_dir).expanduser()
    mat_path = data_dir / f"{dataset}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")

    mat = sio.loadmat(mat_path)
    adj = _first_key(mat, ("Network", "network", "A", "adj", "Adj", "adjacency"))
    features = _to_dense_float_tensor(_first_key(mat, ("Attributes", "attributes", "X", "x", "feature", "features", "attr", "Feat", "feat")))
    labels = _to_label_tensor(_first_key(mat, ("Label", "label", "y", "Y", "gnd", "anomaly_label", "labels")))

    n = features.shape[0]
    if labels.shape[0] != n:
        raise ValueError(f"Feature/label size mismatch: features={n}, labels={labels.shape[0]}")

    edge_index = _edge_index_from_adj(adj)
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=n)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    src, dst = graph.edges()
    edge_index = torch.stack([src, dst], dim=0)

    graph.ndata["feature"] = features
    graph.ndata["label"] = labels

    index_path = data_dir / f"{dataset}_index.txt"
    if index_path.exists():
        index = torch.from_numpy(np.loadtxt(index_path, dtype=np.int64)).long()
    else:
        index = torch.arange(n, dtype=torch.long)
    return graph, features, labels, edge_index, index


def sparse_mx_to_torch_sparse_tensor(sparse_mx) -> torch.Tensor:
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def get_lap(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    edge_indext = edge_index.cpu().numpy().T
    adjacency = sp.csr_matrix(
        (np.ones(edge_indext.shape[0], dtype=np.float32), (edge_indext[:, 0], edge_indext[:, 1])),
        shape=(n, n),
    )
    lap = csgraph.laplacian(adjacency, normed=True)
    return sparse_mx_to_torch_sparse_tensor(lap)


def get_infmatrix(edge_index: torch.Tensor, n: int, m: int, eps: float = 0.0) -> torch.Tensor:
    deg = degree(edge_index[0], n) + 1
    deg = torch.sqrt(deg / (2 * m + n))
    deg = torch.where(deg < eps, torch.zeros_like(deg), deg)
    deg = deg.unsqueeze(dim=-1).to_sparse()
    return torch.sparse.mm(deg, deg.transpose(0, 1)).coalesce()


def compute_metrics(labels: torch.Tensor, scores: torch.Tensor, index: Optional[torch.Tensor] = None) -> tuple[float, float]:
    if index is None:
        index = torch.arange(labels.shape[0])
    y_true = labels[index].detach().cpu().numpy()
    y_score = scores[index].detach().cpu().numpy()
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUC/AUPRC require both normal and anomaly labels in the evaluation index.")
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)
