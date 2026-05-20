"""Pre-computed graph tensors used by SmoothGNN."""

from __future__ import annotations

import copy

import dgl.function as fn
import torch


class GraphData:
    def __init__(self, graph, features, labels, edge_index, infmatrix, lap, hop: int):
        self.graph = graph
        self.features = features
        self.labels = labels
        self.edge_index = edge_index
        self.infmatrix = infmatrix
        self.lap = lap
        self.hop = hop
        self.embeddings: list[torch.Tensor] = []
        self.distances: list[torch.Tensor] = []

        d_inv_sqrt = torch.pow(self.graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1)
        self.xlx = torch.sigmoid(torch.diag(torch.sparse.mm(torch.sparse.mm(self.features.T, self.lap), self.features)))

        tempgraph = copy.deepcopy(self.graph)
        tempinfmatrix = torch.sparse.mm(self.infmatrix, self.features)
        tempembedding = self.features
        self.distances.append(torch.abs(tempembedding - tempinfmatrix))
        self.embeddings.append(tempembedding)

        with tempgraph.local_scope():
            for _ in range(1, hop):
                tempgraph.ndata["h"] = tempembedding * d_inv_sqrt
                tempgraph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                tempembedding = tempgraph.ndata.pop("h") * d_inv_sqrt
                self.distances.append(torch.abs(tempembedding - tempinfmatrix))
                self.embeddings.append(tempembedding)
