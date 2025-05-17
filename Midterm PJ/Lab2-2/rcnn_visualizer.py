import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmengine.structures import InstanceData

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def visualize_two_models(model1_cfg, model1_ckpt, model2_cfg, model2_ckpt, 
                        image_paths, output_dir, score_thr=0.5):
    """
    可视化对比两个模型的最终预测结果（边界框、实例分割掩码、类别标签和得分）
    
    参数:
        model1_cfg: 第一个模型的配置文件路径
        model1_ckpt: 第一个模型的权重文件路径
        model2_cfg: 第二个模型的配置文件路径
        model2_ckpt: 第二个模型的权重文件路径
        image_paths: 要处理的图像路径列表
        output_dir: 输出目录
        score_thr: 显示结果的分数阈值
    """
    # 初始化两个模型
    model1 = init_detector(model1_cfg, model1_ckpt, device='cuda:0')
    model2 = init_detector(model2_cfg, model2_ckpt, device='cuda:0')
    
    # 初始化可视化器
    visualizer1 = VISUALIZERS.build(model1.cfg.visualizer)
    visualizer1.dataset_meta = model1.dataset_meta
    
    visualizer2 = VISUALIZERS.build(model2.cfg.visualizer)
    visualizer2.dataset_meta = model2.dataset_meta
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in image_paths:
        # 检查图像是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图像文件不存在 {img_path}")
            continue
            
        img = mmcv.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        # 获取两个模型的预测结果
        result1 = inference_detector(model1, img)
        result2 = inference_detector(model2, img)
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 可视化第一个模型的结果
        vis_img1 = img.copy()
        visualizer1.add_datasample(
            'result',
            vis_img1,
            data_sample=result1,
            draw_gt=False,
            show=False,
            pred_score_thr=score_thr
        )
        vis_img1 = visualizer1.get_image()
        ax1.imshow(mmcv.bgr2rgb(vis_img1))
        ax1.set_title(f'{os.path.basename(model1_cfg)} (Score > {score_thr})')
        
        # 可视化第二个模型的结果
        vis_img2 = img.copy()
        visualizer2.add_datasample(
            'result',
            vis_img2,
            data_sample=result2,
            draw_gt=False,
            show=False,
            pred_score_thr=score_thr
        )
        vis_img2 = visualizer2.get_image()
        ax2.imshow(mmcv.bgr2rgb(vis_img2))
        ax2.set_title(f'{os.path.basename(model2_cfg)} (Score > {score_thr})')
        
        # 保存对比图
        plt.suptitle(os.path.basename(img_path))
        plt.tight_layout()
        output_path = f'{output_dir}/{os.path.splitext(os.path.basename(img_path))[0]}_compare.jpg'
        plt.savefig(output_path)
        plt.close()
        print(f'已保存对比图: {output_path}')

if __name__ == '__main__':
    # 配置参数（根据实际情况修改）
    mask_rcnn_config = '/mnt/disk2/jiyun.hu/Lab2-2/mask-rcnn_r50_fpn_amp-1x_coco.py'
    mask_rcnn_checkpoint = '/mnt/disk2/jiyun.hu/Lab2-2/work_dirs/mask_rcnn_r50_fpn_amp_voc12/epoch_6.pth'
    
    sparse_rcnn_config = '/mnt/disk2/jiyun.hu/Lab2-2/sparse-rcnn_r50_fpn_1x_coco.py'
    sparse_rcnn_checkpoint = '/mnt/disk2/jiyun.hu/Lab2-2/work_dirs/sparse_rcnn_voc2012/epoch_12.pth'
    
    image_paths = [
        '/mnt/disk2/jiyun.hu/Lab2-2/data/out_images/out1.jpg',
        '/mnt/disk2/jiyun.hu/Lab2-2/data/out_images/out2.jpg',
        '/mnt/disk2/jiyun.hu/Lab2-2/data/out_images/out3.jpg',
    ]
    output_dir = 'model_comparison_results'
    
    visualize_two_models(
        mask_rcnn_config, mask_rcnn_checkpoint,
        sparse_rcnn_config, sparse_rcnn_checkpoint,
        image_paths, output_dir,
        score_thr=0.5  # 置信度阈值
    )