custom_imports = dict(
    imports=['rein.dataset.refuge'],
    allow_failed_imports=False)

refuge_type = "MYREFUGEDataset"
refuge_root = "/home/hkcrc/xymao/Dataset/Optics/REFUGE"
refuge_crop_size = (512, 512)
refuge_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(512, 512)),
    dict(type="RandomCrop", crop_size=refuge_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type='GaussianBlur', sigma=[0.1, 2.0]),
    dict(type='RandomNoise', mean=0, std=10),
    dict(type="PackSegInputs"),
]
refuge_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_refuge = dict(
    type=refuge_type,
    data_root=refuge_root,
    data_prefix=dict(
        img_path="train/Images",
        seg_map_path="train/Masks",
    ),
    # img_suffix='.jpg',  # Verify file extensions
    # seg_map_suffix='.png',
    pipeline=refuge_train_pipeline,
)
val_refuge = dict(
    type=refuge_type,
    data_root=refuge_root,
    data_prefix=dict(
        img_path="val/Images",
        seg_map_path="val/Masks",
    ),
    # img_suffix='.jpg',  # Verify file extensions
    # seg_map_suffix='.png',
    pipeline=refuge_test_pipeline,
)
test_refuge = dict(
    type=refuge_type,
    data_root=refuge_root,
    data_prefix=dict(
        img_path="test/Images",
        seg_map_path="test/Masks",
    ),
    # img_suffix='.jpg',  # Verify file extensions
    # seg_map_suffix='.png',
    pipeline=refuge_test_pipeline,
)
train_dataloader=dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=train_refuge,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_refuge,
)
val_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mIoU", 'mDice'],
)
test_dataloader=dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=test_refuge,
)
test_evaluator=val_evaluator
