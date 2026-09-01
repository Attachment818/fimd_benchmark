# FIMD 评价与可视化协议

## 已确认的数据事实

- 数据集包含 `01_r_t` 至 `70_r_t` 共 70 对。
- 每对包含 `{id}_r.jpg`、`{id}_t.jpg` 和
  `control_points_{id}_r_t.txt`。
- 每个控制点文件包含 12 行、4 列。
- `t` 作为 query/moving，`r` 作为 reference/fixed。
- 控制点第 1、2 列属于 reference；第 3、4 列属于 query。
- 预测变换方向为 query 到 reference。
- 每对 MLE 是 12 个变换后 query 控制点与 reference 控制点之间欧氏距离的平均值。

## 当前工作协议

为兼容不同原始分辨率，先将 reference 图像缩放到 query 的宽高，reference
控制点按相同的 x/y 比例缩放。匹配、变换估计、图像 warp、MLE 和可视化均在
该 query 尺寸坐标空间完成。

统一 overlay 使用缩放后的 reference 与变换后的 query，各占 50% 权重。绿色圆圈
表示 reference GT，红色圆圈表示变换后的 query GT。未配准面板使用 identity
transform，但仍执行相同的 reference resize。

## 尚待论文最终冻结

- FIMD AUC 的积分阈值和离散/连续实现。
- 主表是否统一使用二阶多项式，或保留方法原生几何后端。
- 配准失败在 MLE 汇总中的呈现方式；AUC/成功率必须保留全部 70 对为分母。
- 主表时间是否包含模型加载和文件写入。

## SIFT 协议冒烟测试（2026-09-01）

固定配置使用绿色通道、CLAHE、最多 5000 个 SIFT 特征、0.75 ratio test、
5 px 单应性 RANSAC 预过滤，并在内点上拟合 query 到 reference 的单个二阶
多项式。FIMD02 得到 188 个匹配、98 个内点、MLE 4.073 px；FIMD68 得到
77 个匹配、27 个内点、MLE 19.140 px。两对图像均通过方向、坐标映射和
畸形目视检查。该配置作为公共链路验证结果，尚未替代完整 70 对正式评估。

## 可追溯性要求

每个方法保存代码和权重来源、commit、校验值、环境、运行配置、逐对结果、匹配点、
内点、变换、aligned image、可视化、失败原因和分阶段计时。

## 数据路径

- 本地审计路径：`D:/ComputerCV/fimd_benchmark/data/FIMD`
- 服务器运行路径：`/home/data1/zhangjunhong/fimd_benchmark/datasets/FIMD`

适配器必须通过 `--data-root` 或配置文件接收路径，不在代码中硬编码本地或服务器路径。
