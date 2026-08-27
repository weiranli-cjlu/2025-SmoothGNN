"""Command-line runner for SmoothGNN on GAD .mat datasets."""
from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import torch
from tqdm import tqdm

from config import get_default_config
from graphdata import GraphData
from model import NAD
from utils import compute_metrics, get_infmatrix, get_lap, load_data, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmoothGNN for unsupervised node anomaly detection")
    parser.add_argument("--dataset", "--data", dest="dataset", default="Amazon", help="Dataset name without .mat")
    parser.add_argument("--data_dir", default="~/datasets/GAD/mat", help="Directory containing .mat datasets")
    parser.add_argument("--result_csv", type=str, default=None, help="CSV path for summarized results")
    parser.add_argument(
        "--result-dir",
        "--result_dir",
        dest="result_dir",
        type=str,
        default=None,
        help="Directory for per-trial anomaly scores and ground-truth labels",
    )
    parser.add_argument("--n_trials", type=int, default=1, help="Number of independent trials")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed; trial i uses seed+i")
    parser.add_argument("--use_original_defaults", action="store_true", help="Use grouped defaults from the original paper/code when available")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--nepoch", "--epochs", dest="nepoch", type=int, default=100, help="Training epochs per trial")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--hop", type=int, default=None, help="K-hop smoothing depth")
    parser.add_argument("--eps", type=float, default=None, help="Epsilon threshold")
    parser.add_argument("--decay", "--weight_decay", dest="decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--init", type=float, default=None, help="Weight initialization std")
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:0")
    parser.add_argument("--verbose", action="store_true", help="Print per-trial progress summary; no per-epoch logs are printed")
    parser.add_argument("--tqdm", action="store_true", help="Show tqdm bar.")
    return parser.parse_args()


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cfg = get_default_config(args.dataset) if args.use_original_defaults else {}
    args.lr = args.lr if args.lr is not None else cfg.get("lr", 1e-4)
    args.hop = args.hop if args.hop is not None else cfg.get("hop", 6)
    args.eps = args.eps if args.eps is not None else cfg.get("eps", 4e-3)
    args.init = args.init if args.init is not None else cfg.get("init", 1e-3)
    if args.use_original_defaults and args.seed == 1 and "seed" in cfg:
        args.seed = cfg["seed"]
    return args


def _safe_filename_component(value: str) -> str:
    """Return a cross-platform-safe filename component."""
    component = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value))
    component = component.strip(" .")
    return component or "unknown"


def save_trial_results(
    result_dir: str | Path,
    dataset: str,
    trial_id: int,
    seed: int,
    scores,
    labels,
) -> Path:
    """Overwrite one trial's per-node anomaly scores and true labels as CSV."""
    score_tensor = torch.as_tensor(scores).detach().cpu().reshape(-1)
    label_tensor = torch.as_tensor(labels).detach().cpu().reshape(-1)

    if score_tensor.numel() != label_tensor.numel():
        raise ValueError(
            "Cannot save trial results: anomaly scores and labels have different "
            f"lengths ({score_tensor.numel()} != {label_tensor.numel()})."
        )

    unique_labels = set(label_tensor.tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            "Cannot save trial results: is_anomaly labels must be 0 or 1, "
            f"got {sorted(unique_labels)}."
        )

    output_dir = Path(result_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(dataset)
    if dataset_name.lower().endswith(".mat"):
        dataset_name = dataset_name[:-4]
    filename = (
        f"{_safe_filename_component(dataset_name)}__SmoothGNN__"
        f"run-{trial_id + 1}__seed-{seed}.csv"
    )
    output_path = output_dir / filename

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "anomaly_score", "is_anomaly"])
        for sample_index, (score, label) in enumerate(zip(score_tensor, label_tensor)):
            writer.writerow([sample_index, f"{float(score):.17g}", int(label)])

    return output_path


def train_one_trial(args: argparse.Namespace, trial_id: int) -> dict:
    seed = set_seed(args.seed + trial_id)
    features, labels, edge_index, index = load_data(args.dataset, args.data_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    features = features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    index = index.to(device)

    n = features.shape[0]
    m = edge_index.shape[1]
    lap = get_lap(edge_index.cpu(), n).to(device)
    infmatrix = get_infmatrix(edge_index.cpu(), n, m, args.eps).to(device)
    graphdata = GraphData(features, labels, edge_index, infmatrix, lap, args.hop)

    net = NAD(features.shape[1], args.hidden_dim, 2, graphdata, args.init).to(device)
    optimizer = torch.optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=args.decay)

    best_auc = float("-inf")
    best_auprc = float("-inf")
    best_epoch = 0
    best_scores = None
    final_loss = 0.0

    loop = range(args.nepoch)
    if args.tqdm:
        loop = tqdm(loop, desc="Epoch", position=1, leave=False)

    for epoch in loop:
        net.train()
        reconembed, anomalyembed = net()
        loss = torch.mean(reconembed[index]) + torch.mean(anomalyembed[index])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

        net.eval()
        with torch.no_grad():
            _, scores = net()
            auc, auprc = compute_metrics(labels, scores, index)
        if auc > best_auc:
            best_auc = auc
            best_auprc = auprc
            best_epoch = epoch + 1
            best_scores = scores.detach().cpu().clone()

    result_dir = getattr(args, "result_dir", None)
    if result_dir is not None:
        if best_scores is None:
            raise RuntimeError("Cannot save trial results because no training epoch was run.")
        result_path = save_trial_results(
            result_dir=result_dir,
            dataset=args.dataset,
            trial_id=trial_id,
            seed=seed,
            scores=best_scores,
            labels=labels,
        )
        print(f"Saved trial results to: {result_path}")

    return {
        "trial": trial_id + 1,
        "seed": seed,
        "best_epoch": best_epoch,
        "auc": best_auc,
        "auprc": best_auprc,
        "loss": final_loss,
    }


def fmt(values: list[float]) -> str:
    vals = [v * 100 for v in values]
    return f"{mean(vals):.2f}±{pstdev(vals):.2f}({max(vals):.2f})"


def append_csv(args: argparse.Namespace, rows: list[dict]) -> None:
    if args.result_csv is None:
        return
    result_path = Path(args.result_csv).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    exists = result_path.exists()

    epochs = [r["best_epoch"] for r in rows]
    out = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": args.dataset,
        "training_rounds": args.nepoch,
        "n_trials": args.n_trials,
        "auc": fmt([r["auc"] for r in rows]),
        "auprc": fmt([r["auprc"] for r in rows]),
        "best_epoch": f"{mean(epochs):.2f}±{pstdev(epochs):.2f}({max(epochs)})",
    }

    with result_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(out)

    print(
        f"Dataset: {out['dataset']} | Epochs: {out['training_rounds']} | Trials: {out['n_trials']} | "
        f"AUC: {out['auc']} | AUPRC: {out['auprc']} | Best epoch: {out['best_epoch']} | CSV: {result_path}"
    )


def main() -> None:
    args = apply_defaults(parse_args())
    rows = []
    loop = range(args.n_trials)
    if args.tqdm:
        loop = tqdm(loop, desc="Trial", position=0, leave=True)

    for trial_id in loop:
        row = train_one_trial(args, trial_id)
        rows.append(row)
        if args.verbose:
            print(
                f"Trial {row['trial']}/{args.n_trials}: seed={row['seed']}, "
                f"best_epoch={row['best_epoch']}, auc={row['auc'] * 100:.2f}, auprc={row['auprc'] * 100:.2f}"
            )

    append_csv(args, rows)


if __name__ == "__main__":
    main()
