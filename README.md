
Change Detection/
│
├── data/                     # 数据集存放位置（CLCD）
│   └── CLCD/
│       ├── train/
│       │   ├── time1/            # 前时相影像
│       │   ├── time2/            # 后时相影像
│       │   └── label/        # 变化标注（二值图）
│       ├── val/              # 验证集（同理）
│       └── test/             # 测试集（同理）
│
├── models/                   # 模型定义
│   ├── siamunet_conc.py      # FC-Siam-conc 结构
│   ├── siamunet_diff.py      # FC-Siam-diff 结构
│   └── fresunet.py           # FC 结构
|
│
├── utils/                    # 工具函数
│   ├── dataset.py            # 数据读取 & 预处理
│   ├── metrics.py            # F1 / IoU / Precision /Recall
│   ├── losses.py             # 损失函数（交叉熵等）
│   └── output.py             # 预测结果图
|
├── train.py                  # 训练脚本（核心流程）
├── eval.py                   # 验证 / 测试脚本
├── config.py                 # 所有超参数配置
├── checkpoints/              # 模型权重
└── logs/                     # 训练日志
