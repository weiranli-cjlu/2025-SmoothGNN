"""Pre-computed graph tensors used by SmoothGNN.

This version removes the DGL dependency.  K-hop smoothing is implemented with
``torch.sparse.mm`` on the symmetrically-normalized adjacency matrix.
"""
from __future__ import annotations

import torch


class GraphData:
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        edge_index: torch.Tensor,
        infmatrix: torch.Tensor,
        lap: torch.Tensor,
        hop: int,
    ) -> None:
        self.features = features
        self.labels = labels
        self.edge_index = edge_index
        self.infmatrix = infmatrix
        self.lap = lap
        self.hop = hop
        self.embeddings: list[torch.Tensor] = []
        self.distances: list[torch.Tensor] = []

        self.xlx = torch.sigmoid(
            torch.diag(torch.sparse.mm(torch.sparse.mm(self.features.T, self.lap), self.features))
        )

        norm_adj = self._build_dgl_equivalent_norm_adj(edge_index, features.shape[0], features.device)
        tempinfmatrix = torch.sparse.mm(self.infmatrix, self.features)
        tempembedding = self.features

        self.distances.append(torch.abs(tempembedding - tempinfmatrix))
        self.embeddings.append(tempembedding)

        for _ in range(1, hop):
            tempembedding = torch.sparse.mm(norm_adj, tempembedding)
            self.distances.append(torch.abs(tempembedding - tempinfmatrix))
            self.embeddings.append(tempembedding)

    @staticmethod
    def _build_dgl_equivalent_norm_adj(
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the sparse matrix equivalent to DGL update_all normalization.

        Original DGL logic:
            h_src = x_src * deg_in(src)^-0.5
            h_dst = sum_{src -> dst} h_src
            out_dst = h_dst * deg_in(dst)^-0.5

        For sparse.mm, this is represented as:
            A_norm[dst, src] = deg_in(dst)^-0.5 * deg_in(src)^-0.5
        """
        src = edge_index[0].to(device)
        dst = edge_index[1].to(device)
        deg_in = torch.bincount(dst, minlength=num_nodes).float().clamp(min=1)
        deg_inv_sqrt = deg_in.pow(-0.5)
        values = deg_inv_sqrt[dst] * deg_inv_sqrt[src]
        indices = torch.stack([dst, src], dim=0)
        return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()
