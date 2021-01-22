import os
import os.path as osp
import numpy as np
import argparse
from config import cfg
import cv2
import sys
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

import math
import tensorflow as tf
from tfflat.base import Tester
from tfflat.utils import mem_info
from tfflat.mp_utils import MultiProc
from model_fpn import Model

from gen_batch import generate_batch
from dataset import Dataset
from nms.nms import oks_nms

import pdb

num_kps = 8
kps_names = ["l_up", "r_up", "r_down", "l_down", "l_up1", "r_up1", "r_down1", "l_down1" ]

kps_symmetry = [(0, 1), (3, 2), (4,5), (7,6)]
kps_lines = [(0, 1), (1, 2), (2, 3), (3, 0), (4,5), (5,6), (6,7), (7,4)]

def vis_keypoints(img, kps, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    #kps_lines = [(0, 1), (1, 2), (2, 3), (3, 0), (4,5), (5,6), (6,7), (7,4)]
    cnt = 0
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=4, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cnt += 1
            cv2.putText(kp_mask, '%s' % i1, (int(p1[0]), int(p1[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.putText(kp_mask, '%s' % i2, (int(p2[0]), int(p2[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0), cnt

def test_net(tester, val_image, hh_ori, ww_ori):

    imgs = []
    #pdb.set_trace()
    img = cfg.normalize_input(val_image)
    imgs.append(img)
    imgs = np.array(imgs)
    start_time = time.time()
    heatmap = tester.predict_one([imgs])[0] #(1, 64, 64, 4)
    kps_result = np.zeros((1, cfg.num_kps, 3))
    
    if cfg.flip_test:
        flip_imgs = imgs[:, :, ::-1, :]
        flip_heatmap = tester.predict_one([flip_imgs])[0]
       
        flip_heatmap = flip_heatmap[:, :, ::-1, :]
        for (q, w) in cfg.kps_symmetry:
            flip_heatmap_w, flip_heatmap_q = flip_heatmap[:,:,:,w].copy(), flip_heatmap[:,:,:,q].copy()
            flip_heatmap[:,:,:,q], flip_heatmap[:,:,:,w] = flip_heatmap_w, flip_heatmap_q
        flip_heatmap[:,:,1:,:] = flip_heatmap.copy()[:,:,0:-1,:]
        heatmap += flip_heatmap
        heatmap /= 2

    # for each human detection from clustered batch     
    for j in range(cfg.num_kps):
        hm_j = heatmap[0, :, :, j]
        idx = hm_j.argmax()
        y, x = np.unravel_index(idx, hm_j.shape)
        
        px = int(math.floor(x + 0.5))
        py = int(math.floor(y + 0.5))
        if 1 < px < cfg.output_shape[1]-1 and 1 < py < cfg.output_shape[0]-1:
            diff = np.array([hm_j[py][px+1] - hm_j[py][px-1],
                             hm_j[py+1][px]-hm_j[py-1][px]])
            diff = np.sign(diff)
            x += diff[0] * .25
            y += diff[1] * .25
        kps_result[0, j, :2] = (x * cfg.input_shape[1] / cfg.output_shape[1], y * cfg.input_shape[0] / cfg.output_shape[0])
        kps_result[0, j, 2] = hm_j.max() / 255 

    # map back to original images
    for j in range(cfg.num_kps):
        kps_result[0, j, 0] = kps_result[0, j, 0] * ww_ori / cfg.input_shape[1] 
        kps_result[0, j, 1] = kps_result[0, j, 1] * hh_ori / cfg.input_shape[0] 
               
    return kps_result


def test(test_model, test_img):
    
    ori_image = cv2.imread(osp.join('../test_image/', test_img), cv2.IMREAD_COLOR)
    name = test_img.split('.')[0]
    img_path = "../test_vis/{}".format(name)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    hh_ori, ww_ori, _ = ori_image.shape
    # width = height = 256
    [width, height] = [192, 256]
    if width is not None and height is not None:
        val_image = cv2.resize(ori_image, (width, height))

    cfg.set_args(args.gpu_ids.split(',')[0])
    tester = Tester(Model(), cfg)
    tester.load_weights(test_model, test=False)
    kps_result = test_net(tester, val_image, hh_ori, ww_ori)

    #vis
    vis = True
    if vis:
        for i in range(len(kps_result)):
            tmpkps = np.zeros((3,cfg.num_kps))
            tmpkps[:2,:] = kps_result[i, :, :2].transpose(1,0)
            tmpkps[2,:] = kps_result[i, :, 2]
            tmpimg, cnt = vis_keypoints(ori_image, tmpkps)
            kps = tmpkps
            print(cnt)
            if cnt == 4:
                x1, y1 = int(kps[0, 0]), int(kps[1, 0])
                x2, y2 = int(kps[0, 1]), int(kps[1, 1])
                x3, y3 = int(kps[0, 2]), int(kps[1, 2])
                x4, y4 = int(kps[0, 3]), int(kps[1, 3])
                p1 = [x1, y1]
                p2 = [x2, y2]
                p3 = [x3, y3]
                p4 = [x4, y4]
                xmin = min(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)
                xmax = max(x1, x2, x3, x4)
                ymax = max(y1, y2, y3, y4)
                w = xmax - xmin
                h = ymax - ymin
              
                pts_o = np.float32([p1, p2, p3, p4]) # ori
                if w > h:
                    pts_d = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]) # transfer  
                    # get transform matrix
                    M = cv2.getPerspectiveTransform(pts_o, pts_d)
                    # apply transformation
                    dst = cv2.warpPerspective(ori_image, M, (w, h)) 
                else:
                    pts_d = np.float32([[[0, 0], [h, 0], [h, w], [0, w]]]) # transfer  
                    # get transform matrix
                    M = cv2.getPerspectiveTransform(pts_o, pts_d)
                    # apply transformation
                    dst = cv2.warpPerspective(ori_image, M, (h, w)) 

                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(tmpimg)
                plt.subplot(1,2,2)
                plt.imshow(dst)
                plt.show()
                cv2.imwrite(img_path + '/{}.jpg'.format('001'), dst)
           
            if cnt == 8:
                x1, y1 = int(kps[0, 0]), int(kps[1, 0])
                x2, y2 = int(kps[0, 1]), int(kps[1, 1])
                x3, y3 = int(kps[0, 2]), int(kps[1, 2])
                x4, y4 = int(kps[0, 3]), int(kps[1, 3])
                x11, y11 = int(kps[0, 4]), int(kps[1, 4])
                x22, y22 = int(kps[0, 5]), int(kps[1, 5])
                x33, y33 = int(kps[0, 6]), int(kps[1, 6])
                x44, y44 = int(kps[0, 7]), int(kps[1, 7])
                p1, p11 = [x1, y1], [x11, y11]
                p2, p22 = [x2, y2], [x22, y22]
                p3, p33 = [x3, y3], [x33, y33]
                p4, p44 = [x4, y4], [x44, y44]
                xmin, xmin1 = min(x1, x2, x3, x4), min(x11, x22, x33, x44)
                ymin, ymin1 = min(y1, y2, y3, y4), min(y11, y22, y33, y44)
                xmax, xmax1 = max(x1, x2, x3, x4), max(x11, x22, x33, x44)
                ymax, ymax1 = max(y1, y2, y3, y4), max(y11, y22, y33, y44)
                w, w1 = xmax - xmin, xmax1 - xmin1
                h, h1 = ymax - ymin, ymax1 - ymin1
              
                pts_o = np.float32([p1, p2, p3, p4]) # ori
                pts_o1 = np.float32([p11, p22, p33, p44])
                if w > h:
                    pts_d = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]) # transfer
                    pts_d1 = np.float32([[[0, 0], [w1, 0], [w1, h1], [0, h1]]]) # transfer    
                    # get transform matrix
                    M = cv2.getPerspectiveTransform(pts_o, pts_d)
                    M1 = cv2.getPerspectiveTransform(pts_o1, pts_d1)
                    # apply transformation
                    dst = cv2.warpPerspective(ori_image, M, (w, h))
                    dst1 = cv2.warpPerspective(ori_image, M1, (w1, h1))  
                else:
                    pts_d = np.float32([[[0, 0], [h, 0], [h, w], [0, w]]]) # transfer
                    pts_d1 = np.float32([[[0, 0], [h1, 0], [h1, w1], [0, w1]]]) # transfer    
                    # get transform matrix
                    M = cv2.getPerspectiveTransform(pts_o, pts_d)
                    M1 = cv2.getPerspectiveTransform(pts_o1, pts_d1)
                    # apply transformation
                    dst = cv2.warpPerspective(ori_image, M, (h, w)) 
                    dst1 = cv2.warpPerspective(ori_image, M1, (w1, h1)) 

                plt.figure()
                plt.subplot(1,3,1)
                plt.imshow(tmpimg[:,:,[2,1,0]])
                plt.subplot(1,3,2)
                plt.imshow(dst[:,:,[2,1,0]])
                plt.subplot(1,3,3)
                plt.imshow(dst1[:,:,[2,1,0]])
                plt.show()
                cv2.imwrite(img_path + '/{}.jpg'.format('001'), dst)
                cv2.imwrite(img_path + '/{}.jpg'.format('002'), dst1)
                
        #kps = tmpkps
        #print(tmpkps)
        #print(len(kps[0].tolist()), cnt)
        # plt.imshow(tmpimg)
        # plt.show()
        cv2.imwrite(img_path+'/{}.jpg'.format(name), tmpimg)

    
if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, default='1', dest='gpu_ids')
        parser.add_argument('--test_epoch', type=str, default= '180', dest='test_epoch')
        parser.add_argument('--img', type=str, default='9.jpg', dest='test_image') #1292
        args = parser.parse_args()

        # test gpus
        if not args.gpu_ids:
            args.gpu_ids = str(np.argmin(mem_info()))

        if '-' in args.gpu_ids:
            gpus = args.gpu_ids.split('-')
            gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
            gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
            args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
        
        assert args.test_epoch, 'Test epoch is required.'
        return args

    global args
    args = parse_args()

    test(int(args.test_epoch), args.test_image)
