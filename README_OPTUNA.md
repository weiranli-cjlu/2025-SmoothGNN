# SmoothGNN Optuna 调优脚本

## 快速调优

```bash
python tune_optuna.py --dataset Amazon --n_trials 50 --repeat 3 --timeout 7200 --device cuda:0
```

参数含义：

- `--n_trials`：Optuna 搜索次数。
- `--repeat`：每组超参数重复训练次数，目标值取平均 AUC/AUPRC，建议 3；大数据集可设为 1。
- `--timeout`：总调优时间上限，单位秒。
- `--metric auc|auprc`：使用 AUC 或 AUPRC 作为优化目标。
- `--storage sqlite:///results/smoothgnn_optuna.db`：保存 Optuna study，便于中断后继续。
- `--fixed_nepoch 100`：固定训练轮次，不搜索 `nepoch`。
- `--fixed_hidden_dim 64`：固定隐藏维度，不搜索 `hidden_dim`。

## 推荐命令

小数据集：

```bash
python tune_optuna.py --dataset Amazon --n_trials 80 --repeat 3 --timeout 10800 --device cuda:0 \
  --storage sqlite:///optuna_results/smoothgnn_optuna.db
```

中等数据集：

```bash
python tune_optuna.py --dataset YelpChi --n_trials 50 --repeat 3 --fixed_nepoch 150 --device cuda:0 \
  --storage sqlite:///optuna_results/smoothgnn_optuna.db
```

大数据集：

```bash
python tune_optuna.py --dataset elliptic --n_trials 30 --repeat 1 --fixed_hidden_dim 64 --fixed_nepoch 100 --device cuda:0 \
  --storage sqlite:///optuna_results/smoothgnn_optuna.db
```

调优结束后脚本会打印一条 `python main.py ...` 命令，用该命令再做正式 `--n_trials 10` 评估。

## 输出文件

默认追加保存到：

```text
results/smoothgnn_optuna.csv
```

字段包括：时间、数据集、Optuna trial 编号、目标值、AUC、AUPRC、best_epoch、repeat、超参数 JSON。

## 搜索空间

```text
lr:         1e-5 ~ 5e-3, log
weight_decay/decay: 1e-8 ~ 1e-4, log
hop:        2 ~ 8
hidden_dim: 16 / 32 / 64 / 128
eps:        0 / 1e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3 / 8e-3 / 1e-2
init:       1e-4 ~ 8e-2, log
nepoch:     50 ~ 300, step 50
```

这些范围比较保守，主要避免在大图上过多 OOM 或极慢组合。若只调 Amazon、Reddit 等较小数据集，可以扩大 `hidden_dim` 或 `nepoch`。
