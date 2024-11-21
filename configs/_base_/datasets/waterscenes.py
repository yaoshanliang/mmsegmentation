# dataset settings
dataset_type = 'WaterScenesDataset' # 数据集类型，这将被用来定义数据集
data_root = '/data/data1/shanliang/MaCVi/LaRs/WaterScenes/' # 数据的根路径
ignore_idx=255
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024) # 训练时的裁剪大小
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=2, # 每一个GPU的batch size大小
    num_workers=12, # 为每一个GPU预读取数据的进程个数
    persistent_workers=True, # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='InfiniteSampler', shuffle=True), # 训练时进行随机洗牌(shuffle)
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images', seg_map_path='coco/SegmentationClass'), # 训练数据的前缀
        # ann_file='all.txt',
        pipeline=[
            dict(type='LoadImageFromFile'), # 第1个流程，从文件路径里加载图像
            dict(type='LoadAnnotations'), # 第2个流程，对于当前图像，加载它的标注图像
            dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True), # 调整输入图像大小(resize)和其标注图像的数据增广流程
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), # 随机裁剪当前图像和其标注图像的数据增广流程
            dict(type='RandomFlip', prob=0.5), # 翻转图像和其标注图像的数据增广流程
            dict(type='PhotoMetricDistortion'), # 光学上使用一些方法扭曲当前图像和其标注图像的数据增广流程
            dict(type='PackSegInputs')  # 打包用于语义分割的输入数据
            # dict(type='Normalize', **img_norm_cfg)
        ]
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False), # 训练时不进行随机洗牌(shuffle)
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images', seg_map_path='coco/SegmentationClass'),
        ann_file='val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'), # 第1个流程，从文件路径里加载图像
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True), # 使用调整图像大小(resize)增强
            # 在' Resize '之后添加标注图像
            # 不需要做调整图像大小(resize)的数据变换
            dict(type='LoadAnnotations'), # 加载数据集提供的语义分割标注
            dict(type='PackSegInputs') # 打包用于语义分割的输入数据
        ]))

# 精度评估方法，我们在这里使用 IoUMetric 进行评估
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])


test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
    format_only=True,
    output_dir='work_dirs/format_results')
test_dataloader = dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images'),
        ann_file='test.txt',
        # 测试数据变换中没有加载标注
        pipeline=[
            dict(type='LoadImageFromFile'), # 第1个流程，从文件路径里加载图像
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True), # 使用调整图像大小(resize)增强
            dict(type='PackSegInputs') # 打包用于语义分割的输入数据
        ]))
