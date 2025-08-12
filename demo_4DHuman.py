import zmq
from ultralytics import FastSAM
import sys, os
# sys.path.append('/home/phuang/4D-Humans')

from pathlib import Path
import torch
import argparse
import cv2
import numpy as np

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

import open3d as o3d

from util import contact_estimation
import time

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)




def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://localhost:5557")

    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # yolov10 = YOLOv10.from_pretrained(f'jameslahm/yolov10b')
    # yolo11_seg = YOLO('/home/phuang/behave-dataset/checkpoints/yolo11x-seg.pt')
    fast_sam = FastSAM('/home/phuang/behave-dataset/checkpoints/FastSAM-x.pt')

    xy_cam_grid = None

    while True:
        data = socket.recv_pyobj()

        rgb_image = data['rgb_image']
        depth_image = data['depth_image']
        human_bbox = data['human_bbox']
        cam_K = data['cam_K']
        left_hand_2d, right_hand_2d = data['left_hand_2d'], data['right_hand_2d']
        

        seg_res = fast_sam.track(source=rgb_image, bboxes=human_bbox, device="cuda", retina_masks=True, imgsz=1048, conf=0.4, iou=0.9)
        # seg_res = yolo11_seg.track(source=rgb_image, imgsz=1024)

        # print(seg_res)
        human_bbox = np.array(human_bbox)[None, ...]

        # seg_classes = seg_res[0].boxes.cls.cpu().numpy()

        human_mask = seg_res[0].masks.data.cpu().numpy()
        

        # visualize_3d_point_cloud(rgb_image, depth_image, human_mask, cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2])

        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(model_cfg, rgb_image, human_bbox)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_smpls = []
        all_kp3ds = []
        all_kp2ds = []
        
        res_imgs = []
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            batch_size = batch['img'].shape[0]

            pred_smpl_params = out['pred_smpl_params']

            pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
            pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
            pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)

            for n in range(batch_size):

                verts = out['pred_vertices'][n].detach().cpu().numpy()
                kp3ds = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                kp2ds = out['pred_keypoints_2d'][n].detach().cpu().numpy()
                # cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_smpls.append({
                    'global_orient' : pred_smpl_params['global_orient'][n].detach().cpu().numpy(),
                    'body_pose': pred_smpl_params['body_pose'][n].detach().cpu().numpy(),
                    'betas': pred_smpl_params['betas'][n].detach().cpu().numpy(),
                })
                all_kp3ds.append(kp3ds)
                all_kp2ds.append(kp2ds)
               

        if xy_cam_grid is None:
            H, W = depth_image.shape
    
            fx, fy, cx, cy = cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2]
            # Generate 3D coordinates from depth
            x_lin = torch.arange(W, device=device)
            y_lin = torch.arange(H, device=device)
            xy_cam_grid = torch.meshgrid(x_lin, y_lin, indexing='xy')

            # x_grid = (x_grid - cx)/fx
            # y_grid = (y_grid - cy)/fy
            # xy_cam_grid = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], -1).reshape(-1, 3)

        start = time.time()
        human_pnts, other_pnts, contact_pnts = contact_estimation(rgb_image, depth_image, human_mask, cam_K, xy_cam_grid, hand_kpts=(left_hand_2d, right_hand_2d))
        # human_pnts, other_pnts, contact_pnts = None, None, None
        duration = time.time() - start
        # pnts = xy_cam_grid * torch.from_numpy(depth_image).to(device=device).reshape(-1, 1)



        res = {
            'boxes': human_bbox,
            # 'classes': obj_classes,
            # 'ids': obj_ids,
            'human_boxes': human_bbox,
            'human_verts': all_verts,
            'human_kp3ds': all_kp3ds,
            'human_kp2ds': all_kp2ds,
            'smpl_params': all_smpls,
            'human_mask': human_mask,
            'human_pnts': human_pnts,
            'other_pnts': other_pnts, 
            'contact_pnts': contact_pnts,
            'duration' : duration
    
            # 'res_imgs': res_imgs
        }

        socket.send_pyobj(res)

if __name__ == "__main__":
    main()