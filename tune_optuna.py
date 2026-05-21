"""Optuna hyper-parameter tuning for SmoothGNN.

Usage example:
    python tune_optuna.py --dataset Amazon --n_trials 50 --repeat 3 --timeout 7200

After tuning, run SmoothGNN with the printed best command.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import optuna
import torch

# Make sure this script can import main.py when placed in the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import train_one_trial  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune SmoothGNN with Optuna")
    parser.add_argument("--dataset", required=True, help="Dataset name without .mat")
    parser.add_argument("--data_dir", default="~/datasets/GAD/mat")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each Optuna trial with different seeds and optimize mean AUC")
    parser.add_argument("--seed", type=int, default=1, help="Base seed for Optuna and model runs")
    parser.add_argument("--timeout", type=int, default=None, help="Optuna timeout in seconds")
    parser.add_argument("--study_name", default=None, help="Optuna study name")
    parser.add_argument("--storage", default=None, help="Optuna storage, e.g. sqlite:///smoothgnn_optuna.db")
    parser.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--metric", choices=["auc", "auprc"], default="auc", help="Metric used as Optuna objective")
    parser.add_argument("--result_csv", default="results/smoothgnn_optuna.csv", help="CSV path for tuning records")
    parser.add_argument("--fixed_nepoch", type=int, default=None, help="Fix epochs instead of tuning nepoch")
    parser.add_argument("--fixed_hidden_dim", type=int, default=None, help="Fix hidden_dim instead of tuning it")
    parser.add_argument("--use_original_defaults", action="store_true", help="Only affects fallback values when a search space is fixed")
    parser.add_argument("--tqdm", action="store_true", help="Show inner training tqdm bars")
    parser.add_argument("--verbose", action="store_true", help="Print every Optuna trial summary")
    return parser.parse_args()


def suggest_params(trial: optuna.Trial, cli: argparse.Namespace) -> dict:
    """Search space chosen for SmoothGNN's current CLI parameters.

    Ranges are intentionally conservative to avoid many unstable/OOM trials on large GAD datasets.
    Expand them if you tune only small datasets.
    """
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "decay": trial.suggest_float("decay", 1e-8, 1e-4, log=True),
        "hop": trial.suggest_int("hop", 2, 8),
        "eps": trial.suggest_categorical("eps", [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1e-2]),
        "init": trial.suggest_float("init", 1e-4, 8e-2, log=True),
        "hidden_dim": cli.fixed_hidden_dim if cli.fixed_hidden_dim is not None else trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
        "nepoch": cli.fixed_nepoch if cli.fixed_nepoch is not None else trial.suggest_int("nepoch", 50, 300, step=50),
    }
    return params


def build_train_args(cli: argparse.Namespace, params: dict, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=cli.dataset,
        data_dir=cli.data_dir,
        result_csv=cli.result_csv,
        n_trials=1,
        seed=seed,
        use_original_defaults=cli.use_original_defaults,
        lr=float(params["lr"]),
        nepoch=int(params["nepoch"]),
        hidden_dim=int(params["hidden_dim"]),
        hop=int(params["hop"]),
        eps=float(params["eps"]),
        decay=float(params["decay"]),
        init=float(params["init"]),
        device=cli.device,
        verbose=False,
        tqdm=cli.tqdm,
    )


def fmt_metric(values: list[float]) -> str:
    vals = [v * 100 for v in values]
    return f"{mean(vals):.2f}±{pstdev(vals):.2f}（{max(vals):.2f}）"


def append_record(path: str | Path, row: dict) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def objective_factory(cli: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cli)
        aucs: list[float] = []
        auprcs: list[float] = []
        best_epochs: list[int] = []

        for repeat_id in range(cli.repeat):
            run_seed = cli.seed + trial.number * max(cli.repeat, 1) + repeat_id
            train_args = build_train_args(cli, params, run_seed)
            try:
                out = train_one_trial(train_args, 0)
            except RuntimeError as exc:
                # Commonly catches CUDA OOM. Free cache and prune the trial instead of killing the study.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                trial.set_user_attr("failed_reason", repr(exc))
                raise optuna.TrialPruned(f"training failed: {exc}") from exc

            aucs.append(float(out["auc"]))
            auprcs.append(float(out["auprc"]))
            best_epochs.append(int(out["best_epoch"]))

            score_so_far = mean(aucs if cli.metric == "auc" else auprcs)
            trial.report(score_so_far, step=repeat_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        value = mean(aucs if cli.metric == "auc" else auprcs)
        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dataset": cli.dataset,
            "trial_number": trial.number,
            "objective": f"{value * 100:.2f}",
            "auc": fmt_metric(aucs),
            "auprc": fmt_metric(auprcs),
            "best_epoch": f"{mean(best_epochs):.2f}±{pstdev(best_epochs):.2f}（{max(best_epochs)}）",
            "repeat": cli.repeat,
            "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        }
        append_record(cli.result_csv, row)

        if cli.verbose:
            print(f"trial={trial.number} value={value * 100:.2f} params={params}")
        return value

    return objective


def main() -> None:
    cli = parse_args()
    sampler = optuna.samplers.TPESampler(seed=cli.seed, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)
    study_name = cli.study_name or f"smoothgnn_{cli.dataset}_{cli.metric}"
    study = optuna.create_study(
        study_name=study_name,
        storage=cli.storage,
        direction=cli.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective_factory(cli), n_trials=cli.n_trials, timeout=cli.timeout, gc_after_trial=True, show_progress_bar=True)

    best = study.best_trial
    print("\nBest trial")
    print(f"  number: {best.number}")
    print(f"  {cli.metric}: {best.value * 100:.2f}")
    print(f"  params: {json.dumps(best.params, ensure_ascii=False, sort_keys=True)}")

    p = best.params
    command = (
        f"python main.py --dataset {cli.dataset} --data_dir {cli.data_dir} "
        f"--n_trials 10 --seed {cli.seed} --lr {p['lr']} --weight_decay {p['decay']} "
        f"--hidden_dim {p['hidden_dim']} --hop {p['hop']} --eps {p['eps']} "
        f"--init {p['init']} --nepoch {p['nepoch']} --device {cli.device}"
    )
    print("\nRecommended final evaluation command")
    print(f"  {command}")


if __name__ == "__main__":
    main()
