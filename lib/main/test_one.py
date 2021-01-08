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
from model import Model

from gen_batch import generate_batch
from dataset import Dataset
from nms.nms import oks_nms

import pdb
       
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
    hh_ori, ww_ori, _ = ori_image.shape
    # width = height = 256
    [width, height] = [192, 256]
    if width is not None and height is not None:
        val_image = cv2.resize(ori_image, (width, height))

    cfg.set_args(args.gpu_ids.split(',')[0])
    tester = Tester(Model(), cfg)
    tester.load_weights(test_model, test=True)
    kps_result = test_net(tester, val_image, hh_ori, ww_ori)

    #vis
    vis = True
    if vis:
        for i in range(len(kps_result)):
            tmpkps = np.zeros((3,cfg.num_kps))
            tmpkps[:2,:] = kps_result[i, :, :2].transpose(1,0)
            tmpkps[2,:] = kps_result[i, :, 2]
            tmpimg = cfg.vis_keypoints(ori_image, tmpkps)
        plt.imshow(tmpimg)
        plt.show()
        cv2.imwrite('../test_vis/283.jpg', tmpimg)

    
if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, default='1', dest='gpu_ids')
        parser.add_argument('--test_epoch', type=str, default= '180', dest='test_epoch')
        parser.add_argument('--img', type=str, default='283.jpg', dest='test_image')
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
