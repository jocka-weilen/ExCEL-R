# CoSeR-CLIP 实验协议（AAAI-27）

## 1. 主实验

| 数据集 | 训练迭代 | 输入 | Backbone | 评测尺度 | CAM mIoU | Seg mIoU |
|---|---:|---:|---|---|---:|---:|
| PASCAL VOC 2012 val | 30,000 | 320×320 | CLIP ViT-B/16 | 0.75/1.0/1.25/1.5 + flip | 待完整训练 | 待完整训练 |
| MS COCO 2014 val | 100,000 | 320×320 | CLIP ViT-B/16 | 0.75/1.0/1.25/1.5 + flip | 待完整训练 | 待完整训练 |

本仓库不包含 VOC/COCO 图像和像素标签，因此代码交付阶段不伪造主实验数值。运行 `experiments/run_main_voc.sh` 或 `experiments/run_main_coco.sh` 后，评测程序会生成带完整逐类指标的 JSON 文件。

### 已完成的真实数据冒烟实验

使用本机 VOC2012 的真实图像执行了 1 次 CPU 优化步骤（64×64 工程测试裁剪，batch size 1，完整路由开启）。该测试成功完成数据读取、冻结 CLIP 前向、全部五类损失、反向传播、优化器更新和 checkpoint 保存：

| total | classification | CAM route | online mask | structure | region |
|---:|---:|---:|---:|---:|---:|
| 9.2971 | 2.9788 | 3.1874 | 3.1220 | 0.0000 | 0.1785 |

这些数值只证明工程链路可执行，不是收敛结果，也不能作为论文性能比较。

## 2. 核心消融

所有消融只改变 `--ablation`，其余训练设置、随机种子和评测协议完全一致。

| 配置 | 深层锚点 | 区域所有权 | 混淆竞争 | 浅层结构 | 负向路由 | Seg mIoU |
|---|:---:|:---:|:---:|:---:|:---:|---:|
| `deep_only` | ✓ |  |  |  |  | 待运行 |
| `no_confusion` | ✓ | ✓ |  | ✓ |  | 待运行 |
| `no_structure` | ✓ | ✓ | ✓ |  | ✓ | 待运行 |
| `no_negative_routing` | ✓ | ✓ | ✓ | ✓ |  | 待运行 |
| `full` | ✓ | ✓ | ✓ | ✓ | ✓ | 待运行 |

建议至少使用 3 个随机种子（0、1、2），报告均值与标准差：

```bash
for seed in 0 1 2; do
  VOC_ROOT=/path/to/VOC2012 \
    bash experiments/run_ablation_voc.sh --seed "$seed" --amp
done
```

## 3. 关键超参数实验

在完整模型基础上分别考察：

1. 混淆类别数 `topk_confusions ∈ {1, 3, 5}`；
2. 区域查询数 `num_region_queries ∈ {8, 12, 16}`；
3. 证据预热 `warmup_iters ∈ {0, 500, 1000, 2000}`；
4. 混淆线索：仅文本、仅视觉、仅重叠，以及三者等权；
5. 在线标签阈值 `(tau_b, tau_f)`：`(0.15,0.50)`、`(0.20,0.55)`、`(0.25,0.60)`。

示例：

```bash
bash run_train.sh voc \
  --data_folder /path/to/VOC2012 \
  --topk_confusions 5 \
  --num_region_queries 16 \
  --log_tag k5_q16 \
  --amp
```

## 4. 复现记录

每次运行目录包含：

- `config.json`：完整配置；
- `train.log`：各损失、学习率、验证表；
- `checkpoints/coser_clip_iter_*.pth`：中间模型；
- `checkpoints/coser_clip_final.pth`：最终模型；
- `*.metrics.json`：多尺度评测结果。

提交论文表格前，应固定代码 commit、数据版本、GPU 型号、PyTorch/CUDA 版本、随机种子和 checkpoint SHA-256。
