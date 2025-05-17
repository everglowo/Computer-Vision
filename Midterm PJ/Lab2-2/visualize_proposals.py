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

def visualize_comparison(config_file, checkpoint_file, image_paths, output_dir, topk=100, score_thr=0.3):
    """可视化对比RPN proposals和最终预测结果（包含实例分割掩模）"""
    # 初始化模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    # 初始化可视化器
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in image_paths:
        # 检查图像是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图像文件不存在 {img_path}")
            continue
            
        img = mmcv.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        # 1. 获取RPN proposals（第一阶段）
        data = {
            'inputs': [torch.from_numpy(img).permute(2, 0, 1).float().cuda()],
            'data_samples': [InstanceData(metainfo={
                'img_id': 0,
                'img_shape': (img_h, img_w),
                'scale_factor': np.array([1.0, 1.0, 1.0, 1.0]),
                'ori_shape': (img_h, img_w)
            })]
        }
        
        with torch.no_grad():
            # 预处理数据
            data = model.data_preprocessor(data, False)
            
            # 提取特征
            feats = model.extract_feat(data['inputs'])
            
            # 获取RPN proposals
            rpn_results = model.rpn_head.predict(feats, data['data_samples'], rescale=False)
        
        # 2. 获取最终预测结果（包含掩模）
        final_results = inference_detector(model, img_path)
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # ------------------------- 绘制RPN proposals（蓝色框）-------------------------
        ax1.imshow(mmcv.bgr2rgb(img))
        for i, proposal in enumerate(rpn_results[0].bboxes.cpu().numpy()[:topk]):
            x1, y1, x2, y2 = map(int, proposal)
            ax1.add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1, fill=False, 
                edgecolor='blue', linewidth=0.8, alpha=0.7))
        ax1.set_title(f'RPN Proposals (Top {topk})')
        
        # ------------------------- 绘制最终预测结果（使用可视化器显示掩模）-------------------------
        # 创建一个新的可视化图像
        vis_img = img.copy()
        visualizer.add_datasample(
            'result',
            vis_img,
            data_sample=final_results,
            draw_gt=False,
            show=False,
            pred_score_thr=score_thr
        )
        vis_result = visualizer.get_image()
        
        # 显示带有掩模的结果
        ax2.imshow(mmcv.bgr2rgb(vis_result))
        ax2.set_title(f'Final Predictions with Masks (Score > {score_thr})')
        
        # 保存对比图
        plt.suptitle(os.path.basename(img_path))
        plt.tight_layout()
        output_path = f'{output_dir}/{os.path.splitext(os.path.basename(img_path))[0]}_compare.jpg'
        plt.savefig(output_path)
        plt.close()
        print(f'已保存对比图: {output_path}')

if __name__ == '__main__':
    # 配置参数（根据实际情况修改）
    config_file = '/mnt/disk2/jiyun.hu/Lab2-2/mask-rcnn_r50_fpn_amp-1x_coco.py'
    checkpoint_file = '/mnt/disk2/jiyun.hu/Lab2-2/work_dirs/mask_rcnn_r50_fpn_amp_voc12/epoch_6.pth'
    image_paths = [
        '/mnt/disk2/jiyun.hu/Lab2-2/data/test_images/test1.jpg',
        '/mnt/disk2/jiyun.hu/Lab2-2/data/test_images/test2.jpg',
        '/mnt/disk2/jiyun.hu/Lab2-2/data/test_images/test3.jpg',
    ]
    output_dir = 'mask_visualization_results'
    
    visualize_comparison(config_file, checkpoint_file, image_paths, output_dir)