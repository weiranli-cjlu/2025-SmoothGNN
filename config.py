"""Default command-line configurations for SmoothGNN."""

from __future__ import annotations

# Paper/original-code style grouped defaults converted to CLI defaults.
# Users can override every value from the command line.
DEFAULT_CONFIGS = {
    "small": {"lr": 1e-4, "hop": 4, "init": 0.05, "seed": 97, "eps": 0.0},
    "medium": {"lr": 5e-4, "hop": 5, "init": 0.01, "seed": 43, "eps": 4e-3},
    "large": {"lr": 5e-4, "hop": 6, "init": 0.05, "seed": 23, "eps": 4e-3},
}

DATASET_GROUPS = {
    "Reddit": "small",
    "reddit": "small",
    "tolokers": "small",
    "Amazon": "small",
    "Amazon-all": "small",
    "YelpChi": "medium",
    "YelpChi-all": "medium",
    "questions": "medium",
    "t_finance": "medium",
    "elliptic": "large",
}


def get_default_config(dataset: str) -> dict:
    """Return grouped defaults for a dataset, or an empty dict if unknown."""
    group = DATASET_GROUPS.get(dataset)
    if group is None:
        return {}
    return DEFAULT_CONFIGS[group].copy()
