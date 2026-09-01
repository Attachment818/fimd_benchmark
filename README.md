# FIMD 视网膜图像配准基准

本仓库用于独立复现和补充 FIMD 上的视网膜图像配准实验，与
`sr_project` 隔离。原始数据、模型权重、第三方仓库和大规模运行输出不提交 Git。

## 当前目标

1. 固定 70 对 FIMD 的输入方向、缩放、控制点和评价协议。
2. 在 FIMD02 上补齐论文所需方法的统一 overlay 可视化。
3. 在统一协议下运行 70 对数据，保存逐对成功/失败、MLE、变换和时间。

## 快速开始

生成并校验数据清单：

```powershell
python scripts/audit_fimd_dataset.py --data-root data/FIMD --output configs/protocol/pair_manifest.csv
```

运行 SIFT 单对冒烟测试：

```powershell
python -m adapters.run_sift_fimd --data-root data/FIMD --pair-id 02 --output-root outputs/SIFT
```

所有正式运行必须使用唯一运行目录，不覆盖已有结果。
