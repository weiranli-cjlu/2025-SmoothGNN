"""SmoothGNN model."""

from __future__ import annotations

import torch
import torch.nn as nn


class NAD(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_class: int, graphdata, init: float):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.graphdata = graphdata

        # The original implementation used plain Python lists, so parameters were
        # not registered by PyTorch. ModuleList is required for correct training.
        self.featuretrans1 = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(graphdata.hop)])
        self.featuretrans2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(graphdata.hop)])
        self.convs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(graphdata.hop)])
        self.coef = nn.Linear(in_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim * graphdata.hop, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, in_dim)
        self.act = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.init_weight(init)

    def init_weight(self, init: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self):
        device = self.graphdata.features.device
        n = self.graphdata.features.shape[0]
        h1_final = torch.empty((n, 0), device=device)
        h2_final = torch.empty((n, 0), device=device)

        for i, conv in enumerate(self.convs):
            h1 = self.act(self.featuretrans1[i](self.graphdata.embeddings[i]))
            h1 = self.act(self.featuretrans2[i](h1))
            h1 = self.act(conv(h1))
            h1_final = torch.cat([h1_final, h1], dim=-1)

            h2 = self.act(self.featuretrans1[i](self.graphdata.distances[i]))
            h2 = self.act(self.featuretrans2[i](h2))
            h2 = self.act(conv(h2))
            h2_final = torch.cat([h2_final, h2], dim=-1)

        coef = self.coef(self.graphdata.xlx)
        h1 = self.act(self.linear1(h1_final))
        reconembed = torch.abs(self.linear2(h1 * coef) - self.graphdata.features)

        h2 = self.act(self.linear1(h2_final))
        h2 = self.bn2(h2)
        anomalyembed = torch.mean(torch.sigmoid(torch.mean(h2 * coef, dim=1, keepdim=True)), dim=1)
        return reconembed, anomalyembed
