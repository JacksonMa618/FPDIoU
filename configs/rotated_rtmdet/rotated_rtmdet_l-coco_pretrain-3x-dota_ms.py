_base_ = './rotated_rtmdet_l-3x-dota_ms.py'

coco_ckpt = 'rotated_rtmdet_l-coco_pretrain-3x-dota_ms-06d248a2.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=coco_ckpt)),
    neck=dict(
        init_cfg=dict(type='Pretrained', prefix='neck.',
                      checkpoint=coco_ckpt)),
    bbox_head=dict(
        init_cfg=dict(
            type='Pretrained', prefix='bbox_head.', checkpoint=coco_ckpt)))

# batch_size = (2 GPUs) x (4 samples per GPU) = 8
train_dataloader = dict(batch_size=4, num_workers=4)
