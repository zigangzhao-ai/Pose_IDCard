#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import glob
import numpy as np
import cv2
import json
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat

import sys
cur_dir = os.path.dirname(__file__)
sys.path.insert(0, osp.join(cur_dir, 'PythonAPI'))
from pycocotools.coco import COCO

class Dataset(object):
    
    dataset_name = 'MPII'
    
    num_kps = 4
    kps_names = ["l_up", "r_up", "r_down", "l_down"]
   

    kps_symmetry = [(0, 1), (3, 2)]
    kps_lines = [(0, 1), (1, 2), (2, 3), (3, 0)]

    img_path = "/workspace/zigangzhao/Pose_IDCard/data/MPII/images/"  
    train_annot_path = "/workspace/zigangzhao/Pose_IDCard/data/MPII/annnotations/train.json"
    test_annot_path = "/workspace/zigangzhao/Pose_IDCard/data/MPII/annnotations/test.json"

    def load_train_data(self, score=False):
        coco = COCO(self.train_annot_path)
        train_data = []
        for aid in coco.anns.keys():
            ann = coco.anns[aid]
            imgname = coco.imgs[ann['image_id']]['file_name']
            joints = ann['keypoints']

            if (ann['image_id'] not in coco.imgs) or ann['iscrowd'] or (ann['num_keypoints'] == 0):
                continue

            # sanitize bboxes
            x, y, w, h = ann['bbox']
            img = coco.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
                bbox = [x1, y1, x2-x1, y2-y1]
            else:
                continue
            
            if score:
                data = dict(image_id = ann['image_id'], imgpath = imgname, bbox=bbox, joints=joints, score=1)
            else:
                data = dict(image_id = ann['image_id'], imgpath = imgname, bbox=bbox, joints=joints)
            train_data.append(data)

        return train_data
        
    def load_val_data_with_annot(self): 
        coco = COCO(self.test_annot_path)
        val_data = []
        for aid in coco.anns.keys():
            ann = coco.anns[aid]
            if ann['image_id'] not in coco.imgs:
                continue
            imgname = coco.imgs[ann['image_id']]['file_name']
            bbox = ann['bbox']
            joints = ann['keypoints']
            data = dict(image_id = ann['image_id'], imgpath = imgname, bbox=bbox, joints=joints, score=1)
            val_data.append(data)
        return val_data

    
    def load_annot(self, db_set):
        if db_set == 'train':
            coco = COCO(self.train_annot_path)
        elif db_set == 'test':
            coco = COCO(self.test_annot_path)
        else:
            print('Unknown db_set')
            assert 0

        return coco

    def load_imgid(self, annot):
        return annot.imgs

    def imgid_to_imgname(self, annot, imgid, db_set):
        imgs = annot.loadImgs(imgid)
        imgname = [i['file_name'] for i in imgs]
        return imgname

    def evaluation(self,result,annot, result_dir, db_set):
        '''
        result_path = osp.join(result_dir, 'result.mat')
        for t in result:
            savemat(result_path, mdict=t)
        '''
        result_path = osp.join(result_dir, 'result.json')
        with open(result_path, 'w') as app:
            json.dump(result, app, indent=4)

    def vis_keypoints(self, img, kps, kp_thresh=0.4, alpha=1):

        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(self.kps_lines) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # Perform the drawing on a copy of the image, to allow for blending.
        kp_mask = np.copy(img)

        # Draw the keypoints.
        #kps_lines=[(0, 1), (1, 2), (2, 3), (3, 0)]
        for l in range(len(self.kps_lines)):
            i1 = self.kps_lines[l][0]
            i2 = self.kps_lines[l][1]
            p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
            p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
            if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                cv2.line(
                    kp_mask, p1, p2,
                    color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            if kps[2, i1] > kp_thresh:
                cv2.putText(kp_mask, '%s' % i1, (int(p1[0]), int(p1[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.circle(
                    kp_mask, p1,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if kps[2, i2] > kp_thresh:
                cv2.putText(kp_mask, '%s' % i2, (int(p2[0]), int(p2[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.circle(
                    kp_mask, p2,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

        # Blend the keypoints.
        return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

dbcfg = Dataset()
