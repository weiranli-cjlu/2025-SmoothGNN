# SmoothGNN Refactor for GAD `.mat` Datasets

本版本将原始 SmoothGNN 代码整理为命令行实验格式，默认从 `~/datasets/GAD/mat` 读取 `.mat` 数据集，不保存模型和训练日志，只在训练结束后输出汇总，并将结果追加到 CSV。

## 环境建议

```bash
uv venv -p 3.12
uv pip install torch==2.4.0 torch_geometric scikit-learn --torch-backend=cpu
uv pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
```

## 数据集目录

```text
~/datasets/GAD/mat
├── ACM.mat
├── Amazon.mat
├── BlogCatalog.mat
├── Disney.mat
├── Enron.mat
├── Flickr.mat
├── Reddit.mat
├── YelpChi.mat
├── ...
```

`.mat` 文件会自动读取常见字段：

- 图结构：`Network / network / A / adj / adjacency`
- 属性：`Attributes / attributes / X / feature / features / feat`
- 标签：`Label / label / y / gnd / anomaly_label`

若目录下存在 `<dataset>_index.txt`，则使用该索引评估；否则默认对所有节点评估。

## 运行方式

基础运行：

```bash
python main.py --dataset Amazon --n_trials 5 --seed 1 --nepoch 100
```

指定数据集目录和结果文件：

```bash
python main.py \
  --dataset YelpChi \
  --data_dir ~/datasets/GAD/mat \
  --result_csv results/smoothgnn_results.csv \
  --n_trials 10 \
  --seed 42 \
  --nepoch 200 \
  --lr 0.0005 \
  --hidden_dim 64 \
  --hop 5 \
  --eps 0.004 \
  --init 0.01 \
  --weight_decay 1e-6
```

使用原作者分组默认配置，并允许继续覆盖其中参数：

```bash
python main.py --dataset Reddit --use_original_defaults --n_trials 5
python main.py --dataset YelpChi --use_original_defaults --n_trials 5 --lr 0.001
python main.py --dataset elliptic --use_original_defaults --n_trials 3 --nepoch 100
```

## 原作者配置到命令行的对应关系

原代码中 `name.py` 的分组配置为：

| group | lr | hop | init | seed | eps |
|---|---:|---:|---:|---:|---:|
| small | 0.0001 | 4 | 0.05 | 97 | 0 |
| medium | 0.0005 | 5 | 0.01 | 43 | 0.004 |
| large | 0.0005 | 6 | 0.05 | 23 | 0.004 |

对应命令示例：

```bash
# small: Reddit / tolokers / Amazon
python main.py --dataset Amazon --lr 0.0001 --hop 4 --init 0.05 --seed 97 --eps 0 --n_trials 5

# medium: YelpChi / questions / t_finance
python main.py --dataset YelpChi --lr 0.0005 --hop 5 --init 0.01 --seed 43 --eps 0.004 --n_trials 5

# large: elliptic 等
python main.py --dataset elliptic --lr 0.0005 --hop 6 --init 0.05 --seed 23 --eps 0.004 --n_trials 3
```

## CSV 输出格式

每次运行会向 `results/smoothgnn_results.csv` 追加一行：

```text
time,dataset,training_rounds,n_trials,auc,auprc,best_epoch
2026-05-20 10:15,Amazon,100,5,90.21±2.33（91.00）,78.32±1.42（80.10）,56.40±12.03（77）
```

其中 AUC/AUPRC 已乘以 100 并保留两位小数，格式为 `均值±标准差（最大值）`。

## 主要修改点

1. 删除训练过程逐 epoch 的 loss/time 打印，只保留最终汇总；可用 `--verbose` 打印每个 trial 的简短结果。
2. 添加 `--n_trials` 和 `--seed`，第 `i` 个 trial 使用 `seed+i`。
3. 默认使用 `~/datasets/GAD/mat/<dataset>.mat`，不再依赖仓库内 `datasets/` 文件夹。
4. 结果追加到 CSV，不保存模型、checkpoint 或训练日志。
5. 修复原 `model.py` 中普通 Python list 存放 `nn.Linear` 的问题，改为 `nn.ModuleList`，确保参数被优化器正确管理。
6. 清理 `name.py` 依赖，将原配置迁移到 `config.py` 和 README 的命令行说明。
