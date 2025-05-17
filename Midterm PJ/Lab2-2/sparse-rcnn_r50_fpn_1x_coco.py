_base_ = 'mmdetection/configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'

BATCH_SIZE = 4 # num of samples per gpu


num_stages = 6
num_proposals = 100  

model = dict(
    type='SparseRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=20)
            for _ in range(num_stages)
        ],
    )
)

dataset_type = 'CocoDataset'
data_root = 'data/coco/'  

classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    persistent_workers=True,  # 加速数据加载
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        data_root=data_root,
        metainfo=dict(
        classes=classes
        )
    )
)

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        data_root=data_root,
        metainfo=dict(
        classes=classes
        )
    )
)

test_dataloader = dict(
    batch_size=BATCH_SIZE,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test2017/'),
        data_root=data_root,
        metainfo=dict(
        classes=classes
        )
    )
)


# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001),
    clip_grad=dict(max_norm=1, norm_type=2)
)
# 学习率调度（缩短热身期）
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),  # 热身迭代从1000减少到500
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[8,11],  # 学习率衰减节点
        gamma=0.1)
]


train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,  
    val_interval=1   # 每个epoch验证一次
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1))

# TensorBoard日志
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

work_dir = 'work_dirs/sparse_rcnn_voc2012'


