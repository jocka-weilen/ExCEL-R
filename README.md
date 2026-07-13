# CoSeR-CLIP

当前发布版本：**v11**。

CoSeR-CLIP 是在 ExCEL 弱监督语义分割代码库上迭代得到的、面向类别共现与语义混淆的 CLIP 有符号证据路由方法。当前实现对应方法稿 `CoSeR_CLIP_Method_v11_CN_AAAI27.tex`，方法名在新增模型、训练、评测、日志和 checkpoint 中统一为 **CoSeR-CLIP**。

> ExCEL 仅作为本仓库的 baseline 和代码来源保留；CoSeR-CLIP 不再使用 ExCEL 的属性文本 CAM 作为主训练路径。

## 方法与代码对应

| 方法组成 | 代码位置 | 实现内容 |
|---|---|---|
| 深层语义锚定 | `model/coser_core.py` | 可训练的深层/全局 CLIP 投影适配器、深层响应和图像级分类 |
| 混淆感知区域所有权 | `model/coser_core.py` | 可学习区域查询、外观—空间区域图、Top-K 文本/视觉/重叠混淆挖掘、加权负类支持 |
| 支持与矛盾证据 | `model/coser_core.py` | 八邻域浅层结构、冻结浅层语义探针、平滑 OR、正负门控 |
| 有符号证据路由 | `model/coser_core.py` | 类别共享两层 1×1 路由器、正证据 Softmax、独立负证据门控、校准 CAM |
| 完整训练目标 | `model/losses.py` | 分类、路由 CAM、停止梯度在线掩码、结构排序、区域多样性与查询使用损失 |
| 视觉分割分支 | `model/model_coser_clip.py` | 冻结 CLIP 多层视觉特征融合；分割头不接收 CAM |

训练前 `warmup_iters` 次迭代仅启用正向证据；之后才启用 Top-K 混淆竞争、困难负类路由和负向抑制。负类深层响应停止梯度，深层与中层目标支持使用平滑 OR 融合。

## 环境和数据

先进入已安装 PyTorch 的环境，再安装 `requirements.txt` 中缺少的依赖。默认使用 CLIP ViT-B/16，并从 `~/.cache/clip/ViT-B-16.pt` 读取或由 CLIP 下载权重。

数据目录与原 baseline 兼容：

```text
VOC2012/
├── JPEGImages
├── SegmentationClass
└── SegmentationClassAug

MSCOCO2014/
├── JPEGImages/{train,val}
└── SegmentationClass/{train,val}
```

类别标签列表位于 `datasets/voc` 和 `datasets/coco`。

## 训练

PASCAL VOC：

```bash
bash run_train.sh voc \
  --data_folder /path/to/VOC2012 \
  --log_tag aaai27_main \
  --amp
```

MS COCO：

```bash
bash run_train.sh coco \
  --data_folder /path/to/MSCOCO2014 \
  --log_tag aaai27_main \
  --amp
```

多卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  bash run_train.sh voc --data_folder /path/to/VOC2012 --amp
```

默认主实验协议为 VOC 30k iterations、COCO 100k iterations、输入裁剪 320、batch size 4/GPU。所有参数和损失权重会写入运行目录的 `config.json`；checkpoint 只保存可训练的 CoSeR-CLIP 与分割分支权重，冻结 CLIP 权重在加载时恢复。

## 评测

```bash
bash infer_seg_voc.sh /path/to/coser_clip_final.pth \
  --data_folder /path/to/VOC2012

bash infer_seg_coco.sh /path/to/coser_clip_final.pth \
  --data_folder /path/to/MSCOCO2014
```

评测默认使用 `0.75,1.0,1.25,1.5` 多尺度和水平翻转，报告 CoSeR CAM 与分割预测 mIoU，并在 checkpoint 同目录生成 `*.metrics.json`。

## 消融实验

```bash
VOC_ROOT=/path/to/VOC2012 bash experiments/run_ablation_voc.sh --amp
```

提供五组一致训练协议：

- `deep_only`：仅保留深层语义锚点；
- `no_confusion`：不启用混淆竞争与困难负类；
- `no_structure`：移除浅层结构正证据；
- `no_negative_routing`：移除负向路由项；
- `full`：完整 CoSeR-CLIP。

完整实验矩阵和结果填写位置见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 自检

无需数据集即可运行核心张量、预热、损失和反向传播测试：

```bash
python -m unittest tests.test_coser_clip -v
```

## Baseline 致谢

本项目从 ExCEL（CVPR 2025）代码库迭代，并沿用了其 CLIP 特征提取和视觉分割解码基础。原始工作：*Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation*。
