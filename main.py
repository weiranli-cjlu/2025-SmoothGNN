"""Command-line runner for SmoothGNN on GAD .mat datasets."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
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


def train_one_trial(args: argparse.Namespace, trial_id: int) -> dict:
    seed = set_seed(args.seed + trial_id)
    graph, features, labels, edge_index, index = load_data(args.dataset, args.data_dir)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    index = index.to(device)

    n = features.shape[0]
    m = edge_index.shape[1]
    lap = get_lap(edge_index.cpu(), n).to(device)
    infmatrix = get_infmatrix(edge_index.cpu(), n, m, args.eps).to(device)
    graphdata = GraphData(graph, features, labels, edge_index, infmatrix, lap, args.hop)

    net = NAD(features.shape[1], args.hidden_dim, 2, graphdata, args.init).to(device)
    optimizer = torch.optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=args.decay)

    best_auc = float("-inf")
    best_auprc = float("-inf")
    best_epoch = 0
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
    return f"{mean(vals):.2f}±{pstdev(vals):.2f}（{max(vals):.2f}）"


def append_csv(args: argparse.Namespace, rows: list[dict]) -> None:
    if args.result_csv is None:
        return
    result_path = Path(args.result_csv).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    exists = result_path.exists()
    out = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": args.dataset,
        "training_rounds": args.nepoch,
        "n_trials": args.n_trials,
        "auc": fmt([r["auc"] for r in rows]),
        "auprc": fmt([r["auprc"] for r in rows]),
        "best_epoch": fmt([r["best_epoch"] / 100.0 for r in rows]).replace("±", "±").replace("（", "（"),
    }
    # best_epoch is not a percentage metric; overwrite with compact integer summary.
    epochs = [r["best_epoch"] for r in rows]
    out["best_epoch"] = f"{mean(epochs):.2f}±{pstdev(epochs):.2f}（{max(epochs)}）"

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
                f"best_epoch={row['best_epoch']}, auc={row['auc']*100:.2f}, auprc={row['auprc']*100:.2f}"
            )
    append_csv(args, rows)


if __name__ == "__main__":
    main()
