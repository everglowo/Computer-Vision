
## Task 1
打开 `lab2-1.ipynb` 

### 准备数据集
运行 Prepare Network 单元格之前的所有代码

### 选择基础网络
在 Prepare Network 部分，指定基础模型

### 冻结指定层
指定冻结层（卷积层/全连接层）

### 训练
运行Training 单元格

### 模型权重
从报告中提供的网盘链接下载



## Task 2

### 环境
```
python==3.8 

torch==1.11.0

mmcv-full==2.0.0rc4

mmdet==3.3.0
```
强烈建议mmcv、mmdet的版本与我保持一致，否则运行代码很多报错:(

### 数据准备
下载VOC2012数据集，将其按 8:1:1 的比例划分为训练集、验证集、测试集，并重构为 COCO 格式，放置在data文件夹下，目录结构可参考如下
```
Lab2-2/


├── data/


│   ├── VOCdevkit/


│   │   └── VOC2012/


│   │       ├── Annotations/


│   │       ├── JPEGImages/


│   │       └── ImageSets/


│   └── coco/  # 转换后的 COCO 格式数据


│       ├── annotations/


│       │   ├── instances\_train2017.json


│       │   ├── instances\_val2017.json


│       │   └── instances\_test2017.json


│       ├── train2017/


│       ├── val2017/


│       └── test2017/
```

### 自定义配置文件
在 `Lab2-2/` 目录下创建以下配置文件：
```
Lab2-2/mask-rcnn_r50_fpn_amp-1x_coco.py

Lab2-2/sparse-rcnn_r50_fpn_1x_coco.py
```

编辑配置文件，修改以下关键参数：




```
# 以 mask-rcnn_r50_fpn_amp-1x_coco.py 为例

classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

model = dict(
  roi_head=dict(

      bbox_head=dict(num_classes=20),  # 根据数据集类别数调整

      mask_head=dict(num_classes=20)

)

)

dataset_type = 'CocoDataset'  # 数据集类型

data_root = 'data/coco/'      # 数据根路径

metainfo = dict(

      classes=classes

)

train_dataloader = dict(
    # batch_size=BATCH_SIZE,
    # num_workers=NUM_WORKERS,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root=DATA_ROOT,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        type=DATASET_TYPE,
        metainfo=dict(
            classes=classes
        )
    ),
    sampler=dict(shuffle=True, type='DefaultSampler')
)

# val_dataloader/test_dataloader类似 
```

### 训练模型
```
# 单 GPU 训练
python Lab2-2\mmdetection\tools\train.py  Lab2-2\mask-rcnn_r50_fpn_amp-1x_coco.py

python Lab2-2\mmdetection\tools\train.py  Lab2-2\sparse-rcnn_r50_fpn_1x_coco.py
```
### 训练日志与权重保存路径
```
work_dir='Lab2-2\work_dirs\mask_rcnn_r50_fpn_amp_voc12': 工作目录，用于存储输出结果（如模型权重、日志等）。
work_dir='Lab2-2\work_dirs\sparse_rcnn_voc2012': 工作目录，用于存储输出结果（如模型权重、日志等）。
```

### 单张图片测试
mmdetection中的` image_demo.py ` 用于对单张图像进行可视化。
```
python demo/image_demo.py <图片路径> <配置文件> <模型权重> 
```

### 模型测试
若配置文件路径为` Lab2-2\mask-rcnn_r50_fpn_amp-1x_coco.py` ，训练好的模型检查点文件路径为:` Lab2-2\work_dirs\mask_rcnn_r50_fpn_amp_voc12\epoch_6.pth ` 

```
python Lab2-2\mmdetection\tools\test.py  Lab2-2\mask-rcnn_r50_fpn_amp-1x_coco.py  Lab2-2\work_dirs\mask_rcnn_r50_fpn_amp_voc12\epoch_6.pth
```

### 可视化
对比Mask RCNN第一阶段和第二阶段
```
python Lab2-2\visualize_proposals.py
```

对比Mask RCNN和Sparse RCNN 
```
python Lab2-2\rcnn_visualizer.py
```
