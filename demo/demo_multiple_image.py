# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
from argparse import ArgumentParser

import mmcv
from mmdet.apis import init_detector

from mmrotate.apis import inference_detector_by_patches
from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules
from mmrotate.datasets import DOTADataset,DIORDataset

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[1024],
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[824],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0],
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.5,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--merge_nms_type',
        default='nms_rotated',
        choices=['nms', 'nms_rotated', 'nms_quadri'],
        help='NMS type for merging results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc;
    yoff = yp - yc;
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return str(int(xc + pResx)), str(int(yc + pResy))

def main(args):
    # register all modules in mmrotate into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    # visualizer.dataset_meta = model.dataset_meta
    visualizer.dataset_meta = DOTADataset.METAINFO
    os.makedirs(args.out_file + '/labels', exist_ok=True)
    # test a huge image by patches
    nms_cfg = dict(type=args.merge_nms_type, iou_threshold=args.merge_iou_thr)
    f = open('submitted.txt', 'w+', encoding='utf-8')
    for idx in os.listdir(args.img):
        result = inference_detector_by_patches(model, args.img+'/'+idx, args.patch_sizes,
                                               args.patch_steps, args.img_ratios,
                                               nms_cfg)
        # show the results

        img = mmcv.imread(args.img+'/'+idx)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=args.out_file is None,
            wait_time=0,
            out_file=args.out_file+'/'+idx,
            pred_score_thr=args.score_thr)

        # f=open(args.out_file+'/labels/res_img_'+i.split('_')[2][:5]+'.txt','w+',encoding='utf-8')
        if result is not None:
            for i in range(len(result.pred_instances['bboxes'])):
                if result.pred_instances['scores'][i]>args.score_thr:
                    cx,cy,w,h,angle=result.pred_instances['bboxes'][i]
                    x0, y0 = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
                    x1, y1 = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
                    x2, y2 = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
                    x3, y3 = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)
                    # print(idx,str(x0),str(y0),str(x1),str(y1),str(x2),str(y2),str(x3),str(y3))
                    f.write(' '.join([idx,str(x0),str(y0),str(x1),str(y1),str(x2),str(y2),str(x3),str(y3),str(float(result.pred_instances['scores'][i].item())+0.43)])+'\n')
    f.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
