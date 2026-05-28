=
                    花卉图像分类项目 (Flower Classification)
=
一、项目简介
--------------------------------------------------------------------------------
本项目基于深度学习实现花卉图像的自动分类，采用 EfficientNet-B3 作为骨干网络，
使用 5 折交叉验证训练，并结合 Temperature Scaling 校准与多模型 Logit 平均策略，
对 100 类花卉进行高精度识别。

训练数据集包含约 19,929 张花卉图片，涵盖 100 个类别（category_id 范围：164~1833），
每张图片带有中英文名称标注。


二、项目结构
--------------------------------------------------------------------------------
flower_project/

├── code/                          # 核心代码目录

│   ├── model.py                   # 模型定义（基于 timm 创建模型，支持本地权重加载）
│   ├── train.py                   # 5 折交叉验证训练脚本（含 Temperature Scaling 校准）
│   ├── predict.py                 # 推理脚本（多模型 Logit 平均，输出 CSV）
│   ├── utils.py                   # 工具函数（随机种子、数据增强/预处理）
│   └── requirements.txt           # Python 依赖列表
│
├── model/                         # 模型权重及配置
│   ├── best_model_fold0.pth       # 第 0 折最优模型权重
│   ├── best_model_fold1.pth       # 第 1 折最优模型权重
│   ├── best_model_fold2.pth       # 第 2 折最优模型权重
│   ├── best_model_fold3.pth       # 第 3 折最优模型权重
│   ├── best_model_fold4.pth       # 第 4 折最优模型权重
│   └── config.json                # 模型配置（模型名、类别数、输入尺寸、温度等）
│
├── train_dataset/                 # 训练图片目录（约 10,000+ 张 .jpg 图片）
├── test/                          # 测试图片目录
├── train_labels.csv               # 训练标签文件（filename, category_id, chinese_name, english_name）
├── results/
│   └── submission.csv             # 推理结果输出（filename, category_id, confidence）
│
├── predict.py                     # 根目录推理脚本（与 code/predict.py 相同）
│
└── submission_skeleton/           # 提交模板目录（用于评测平台提交）
    └── code/
        ├── model.py               # 模型定义
        ├── predict.py             # 推理脚本
        ├── train.py               # 训练脚本
        ├── train_kfold.py         # 5 折训练脚本（convnext_tiny 版本）
        ├── split_local_test.py    # 本地数据集划分工具（按类别分层抽样）
        ├── utils.py               # 工具函数
        ├── requirements.txt       # 依赖列表
        └── model/                 # 模型权重及配置（与上级 model/ 相同）


三、环境要求
--------------------------------------------------------------------------------
- Python >= 3.8
- CUDA（推荐，支持 GPU 加速训练与推理）

Python 依赖包：
  torch >= 1.9
  torchvision
  timm
  pandas
  numpy
  pillow
  scikit-learn
  safetensors（可选，用于加载 .safetensors 格式权重）

安装依赖：
  pip install -r code/requirements.txt


四、模型配置
--------------------------------------------------------------------------------
当前训练好的模型配置如下（详见 model/config.json）：

  骨干网络:     EfficientNet-B3
  分类类别数:   100
  输入图片尺寸: 600 x 600
  归一化均值:   [0.485, 0.456, 0.406]（ImageNet 标准）
  归一化标准差: [0.229, 0.224, 0.225]（ImageNet 标准）
  Temperature:  0.8741（校准后平均温度）
  折数:         5 折


五、使用方法
--------------------------------------------------------------------------------

1. 训练模型
   ----------
   从 code/ 目录运行：

   cd code
   python train.py ../train_dataset ../train_labels.csv [可选参数]

   可选参数：
     --model_name   骨干网络名称       默认: efficientnet_b3
     --size         输入图片尺寸        默认: 600
     --batch_size   批次大小           默认: 32
     --epochs       训练轮数           默认: 60
     --lr           学习率             默认: 2e-4
     --wd           权重衰减           默认: 1e-4
     --seed         随机种子           默认: 42
     --folds        交叉验证折数        默认: 5
     --model_path   本地预训练权重路径   默认: None（在线下载）

   训练完成后，模型权重保存在 model/ 目录，配置写入 model/config.json。

2. 推理预测
   ----------
   从 code/ 目录运行：

   cd code
   python predict.py ../test ../results/submission.csv

   也可使用根目录的 predict.py：

   python predict.py test results/submission.csv

   推理结果为 CSV 格式，包含三列：filename, category_id, confidence。


六、关键技术说明
--------------------------------------------------------------------------------

1. 5 折交叉验证（Stratified K-Fold）
   - 使用 StratifiedKFold 保证每折类别分布一致
   - 每折保存验证集上最优模型

2. Temperature Scaling 校准
   - 训练完成后，在验证集上使用 LBFGS 优化器拟合温度参数
   - 校准后的 logits 经过温度缩放后再做 softmax，提升概率估计可靠性

3. 多模型 Logit 平均
   - 推理时依次加载 5 个折的模型，分别计算 logits
   - 对 logits 取平均后再做 softmax，提高预测鲁棒性

4. category_id 映射
   - 原始 category_id 不连续（164~1833），训练时自动映射为 0~99
   - 映射关系保存在 config.json 的 id2label 字段
   - 推理时自动将预测结果反映射回原始 category_id

5. 数据增强
   - 训练阶段：Resize + RandomHorizontalFlip + ColorJitter + Normalize
   - 验证/推理阶段：Resize + Normalize

6. 混合精度训练
   - 使用 torch.cuda.amp 自动混合精度，加速训练并降低显存占用

7. Label Smoothing
   - 使用 CrossEntropyLoss(label_smoothing=0.1) 防止过拟合


七、输出格式
--------------------------------------------------------------------------------
提交文件 submission.csv 格式：

  filename          category_id    confidence
  img_000051.jpg    164            0.9995
  img_000052.jpg    164            0.9993
  ...


八、本地测试工具
--------------------------------------------------------------------------------
使用 split_local_test.py 可将训练集按类别分层划分为训练/验证集：

  python split_local_test.py train_labels.csv --ratio 0.1

  输出文件：
    train_split.csv       — 90% 训练集
    local_test_split.csv  — 10% 验证集

================================================================================
