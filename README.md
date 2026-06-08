# CALVIN LeRobot ACT 任务二复现说明

本项目用于复现实验二：使用 LeRobot 框架中的 ACT 算法，在 CALVIN 数据集上训练视觉-动作策略，并进行跨环境 zero-shot 动作误差评估。

## 1. 实验内容

本代码对应以下三个实验：

1. 基础策略训练：仅使用环境 B，也就是 `splitB`，训练一个 ACT 视觉-动作策略模型。
2. 多环境联合训练：使用环境 A、B、C，也就是 `splitA + splitB + splitC`，训练一个相同结构和超参数的 ACT 模型。
3. Zero-shot 跨环境测试：将上述模型在未见过的环境 D，也就是 `splitD`，上评估动作误差。

评估指标使用动作误差：

- `Action L1 Loss`：归一化动作空间中的 L1 误差。
- `Raw Action L1`：反归一化后的原始动作误差，报告中主要使用该指标。

## 2. 环境配置

推荐使用 Linux + CUDA GPU 环境。实验实际运行环境为：

- GPU：NVIDIA RTX 3090 24GB
- Python：3.10
- PyTorch：2.5.1 + CUDA 12.1
- LeRobot：0.4.4

### 2.1 使用 environment.yml 创建环境

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate calvin-act
```

检查 PyTorch 和 CUDA 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果输出 `True` 并显示 GPU 名称，则环境正常。

### 2.2 主要依赖

主要依赖已经写在 `environment.yml` 中，包括：

- `torch`
- `torchvision`
- `lerobot`
- `pyarrow`
- `pandas`
- `pillow`
- `matplotlib`
- `tqdm`
- `wandb`
- `huggingface_hub`

## 3. 数据准备

本实验使用 Hugging Face 数据集：

```text
xiaoma26/calvin-lerobot
```

### 3.1 下载数据集

推荐目录结构如下：

```text
/root/autodl-tmp/calvin_act_hw3/
  calvin-lerobot/
  task2_act/
  outputs/
```

下载数据集：

```bash
mkdir -p /root/autodl-tmp/calvin_act_hw3
cd /root/autodl-tmp/calvin_act_hw3

huggingface-cli download xiaoma26/calvin-lerobot \
  --repo-type dataset \
  --local-dir calvin-lerobot
```

如果 Hugging Face 连接较慢，可以使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download xiaoma26/calvin-lerobot \
  --repo-type dataset \
  --local-dir calvin-lerobot
```

下载完成后，目录结构应类似：

```text
calvin-lerobot/
  splitA/
    data/
    meta/
  splitB/
    data/
    meta/
  splitC/
    data/
    meta/
  splitD/
    data/
    meta/
```

## 4. 代码准备

将本仓库中的 `task2_act` 文件夹复制到工作目录：

```bash
cd /root/autodl-tmp/calvin_act_hw3
cp -r /path/to/your/repo/task2_act .
cd task2_act
```

如果你直接在仓库根目录运行，也可以进入：

```bash
cd task2_act
```

注意：`run_train_baseB.sh` 和 `run_train_chunk20.sh` 默认使用以下路径：

```bash
DATA=/root/autodl-tmp/calvin_act_hw3/calvin-lerobot
OUT_ROOT=/root/autodl-tmp/calvin_act_hw3/outputs
```

如果你的数据路径不同，请修改脚本开头的 `DATA` 和 `OUT_ROOT`。

## 5. 快速测试

正式训练前，可以先运行 smoke test，检查数据读取、模型前向传播、反向传播和评估是否正常：

```bash
cd /root/autodl-tmp/calvin_act_hw3/task2_act
bash run_smoke.sh
```

如果成功，会在：

```text
/root/autodl-tmp/calvin_act_hw3/outputs/smoke_chunk20/
```

生成小规模测试结果。

## 6. 训练命令

### 6.1 训练基础策略：仅使用环境 B

该实验对应作业要求中的“基础策略训练”。

```bash
cd /root/autodl-tmp/calvin_act_hw3/task2_act
bash run_train_baseB.sh
```

该脚本会完成：

1. 计算 `splitB` 的 state/action 归一化统计量。
2. 使用 `splitB` 训练 ACT 模型。
3. 在 `splitB` 上评估训练内动作误差。
4. 保存 loss 曲线和 checkpoint。

输出目录：

```text
/root/autodl-tmp/calvin_act_hw3/outputs/act_base_splitB/
```

主要输出文件：

```text
checkpoint_final.pt
metrics.csv
eval_splitB.json
loss_curve.png
loss_curve_raw_action_l1.png
```

### 6.2 训练多环境联合模型：使用 A+B+C

该实验对应作业要求中的“多环境联合训练”。

```bash
cd /root/autodl-tmp/calvin_act_hw3/task2_act
bash run_train_chunk20.sh
```

该脚本会完成：

1. 计算 `splitA + splitB + splitC` 的归一化统计量。
2. 使用 `splitA + splitB + splitC` 训练 ACT 模型。
3. 在 `splitD` 上做 zero-shot 动作误差评估。
4. 保存 loss 曲线和 checkpoint。

输出目录：

```text
/root/autodl-tmp/calvin_act_hw3/outputs/act_chunk20/
```

主要输出文件：

```text
checkpoint_final.pt
metrics.csv
eval_splitD.json
loss_curve.png
loss_curve_raw_action_l1.png
```

## 7. 测试命令

### 7.1 测试基础模型在环境 D 上的 zero-shot 动作误差

```bash
cd /root/autodl-tmp/calvin_act_hw3/task2_act

python eval_act.py \
  --checkpoint /root/autodl-tmp/calvin_act_hw3/outputs/act_base_splitB/checkpoint_final.pt \
  --data-root /root/autodl-tmp/calvin_act_hw3/calvin-lerobot \
  --stats /root/autodl-tmp/calvin_act_hw3/outputs/stats_b_full.npz \
  --split splitD \
  --batch-size 32 \
  --num-workers 8 \
  --eval-batches 200 \
  --output /root/autodl-tmp/calvin_act_hw3/outputs/act_base_splitB/eval_splitD.json
```

输出文件：

```text
/root/autodl-tmp/calvin_act_hw3/outputs/act_base_splitB/eval_splitD.json
```

### 7.2 测试联合模型在环境 D 上的 zero-shot 动作误差

```bash
cd /root/autodl-tmp/calvin_act_hw3/task2_act

python eval_act.py \
  --checkpoint /root/autodl-tmp/calvin_act_hw3/outputs/act_chunk20/checkpoint_final.pt \
  --data-root /root/autodl-tmp/calvin_act_hw3/calvin-lerobot \
  --stats /root/autodl-tmp/calvin_act_hw3/outputs/stats_abc_full.npz \
  --split splitD \
  --batch-size 32 \
  --num-workers 8 \
  --eval-batches 200 \
  --output /root/autodl-tmp/calvin_act_hw3/outputs/act_chunk20/eval_splitD.json
```

输出文件：

```text
/root/autodl-tmp/calvin_act_hw3/outputs/act_chunk20/eval_splitD.json
```

## 8. Action Chunking 对照实验

如需复现动作分块机制分析，可以使用相同网络结构，只修改 `--chunk-size`。

示例：训练 `chunk_size = 10` 的联合模型：

```bash
cd /root/autodl-tmp/calvin_act_hw3/task2_act

python train_act.py \
  --data-root /root/autodl-tmp/calvin_act_hw3/calvin-lerobot \
  --stats /root/autodl-tmp/calvin_act_hw3/outputs/stats_abc_full.npz \
  --output-dir /root/autodl-tmp/calvin_act_hw3/outputs/act_chunk10_10k \
  --train-splits splitA splitB splitC \
  --eval-split splitD \
  --chunk-size 10 \
  --image-size 128 \
  --batch-size 32 \
  --num-workers 8 \
  --steps 10000 \
  --samples-per-epoch 100000 \
  --eval-batches 50 \
  --eval-every 1000 \
  --save-every 5000 \
  --amp
```

测试该模型：

```bash
python eval_act.py \
  --checkpoint /root/autodl-tmp/calvin_act_hw3/outputs/act_chunk10_10k/checkpoint_final.pt \
  --data-root /root/autodl-tmp/calvin_act_hw3/calvin-lerobot \
  --stats /root/autodl-tmp/calvin_act_hw3/outputs/stats_abc_full.npz \
  --split splitD \
  --batch-size 32 \
  --num-workers 8 \
  --eval-batches 200 \
  --output /root/autodl-tmp/calvin_act_hw3/outputs/act_chunk10_10k/eval_splitD.json
```

可以把 `--chunk-size 10` 改为 `20` 或 `30`，并相应修改 `--output-dir`。

## 9. 绘图与结果汇总

训练结束后，可以生成汇总图：

```bash
cd /root/autodl-tmp/calvin_act_hw3

python task2_act/plot_task2_summary.py \
  --results-root /root/autodl-tmp/calvin_act_hw3 \
  --output-dir /root/autodl-tmp/calvin_act_hw3/figures
```

生成的图包括：

```text
figures/convergence_train_l1.png
figures/eval_raw_action_l1_curves.png
figures/zero_shot_splitD_action_error.png
figures/action_chunking_splitD.png
```

## 10. WandB Offline 记录

如果需要生成 WandB offline 记录：

```bash
cd /root/autodl-tmp/calvin_act_hw3

python task2_act/log_wandb_offline.py \
  --results-root /root/autodl-tmp/calvin_act_hw3 \
  --wandb-dir /root/autodl-tmp/calvin_act_hw3/wandb_offline
```

之后可以使用：

```bash
wandb sync /root/autodl-tmp/calvin_act_hw3/wandb_offline/wandb/offline-run-*
```

将离线记录同步到 WandB。

## 11. 复现实验中的主要指标

本次实验得到的核心结果如下：

| 模型 | 训练环境 | 测试环境 | Raw Action L1 |
|---|---|---|---:|
| Base ACT | B | B | 0.658640 |
| Base ACT | B | D | 1.266957 |
| Joint ACT | A+B+C | D | 1.024444 |

Action Chunking 对照：

| Chunk Size | 训练环境 | 测试环境 | Raw Action L1 |
|---:|---|---|---:|
| 10 | A+B+C | D | 0.957529 |
| 20 | A+B+C | D | 1.042842 |
| 30 | A+B+C | D | 1.107348 |

结论：

- 多环境联合训练模型在环境 D 上的动作误差低于只用 B 训练的基础模型。
- 在本实验设置下，`chunk_size = 10` 的 zero-shot 动作误差最低。
- 较大的 action chunk 在跨环境视觉偏移下更容易累积预测误差。

## 12. 目录说明

```text
task2_act/
  calvin_act_common.py      # 数据读取、图像解码、归一化等公共函数
  prepare_stats.py          # 计算 state/action 归一化统计量
  train_act.py              # ACT 训练脚本
  eval_act.py               # 动作误差评估脚本
  plot_results.py           # 单个实验曲线绘制
  plot_task2_summary.py     # 多实验对比图绘制
  log_wandb_offline.py      # WandB offline 记录生成
  run_smoke.sh              # 快速测试脚本
  run_train_baseB.sh        # 基础模型训练脚本
  run_train_chunk20.sh      # A+B+C 联合模型训练脚本
```

