'''
code by zzg 2021/03/01
'''
import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import math

from matplotlib import pyplot as plt
from PIL import Image
import pdb
 
##config
input_shape = (256, 192)
[width, height] = [192, 256]
flip_test = True
output_shape = (input_shape[0]//4, input_shape[1]//4)



##vis the keypoints
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
        # print(i1, i2)
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

##covert((1, 64, 48, 8)-->(1, 8, 3))
def convert(output, hh_ori, ww_ori):

    #output.shape#(1, 64, 48, 8)
    heatmap = output
    kps_result = np.zeros((1, num_kps, 3))
    
    # for each human detection from clustered batch     
    for j in range(num_kps):
        hm_j = heatmap[0, :, :, j]
        #print("hm_j_shape", hm_j.shape) ##(64, 48)
        idx = hm_j.argmax()
        #print("idx=", idx)  ##idx= 2787 [0, 64*48)
        y, x = np.unravel_index(idx, hm_j.shape)  ##get the y,x in the hm_j by the index
        
        px = int(math.floor(x + 0.5))
        py = int(math.floor(y + 0.5))
        if 1 < px < output_shape[1]-1 and 1 < py < output_shape[0]-1:
            diff = np.array([hm_j[py][px+1] - hm_j[py][px-1],
                             hm_j[py+1][px] - hm_j[py-1][px]])
            diff = np.sign(diff)
            x += diff[0] * .25
            y += diff[1] * .25
        kps_result[0, j, :2] = (x * input_shape[1] / output_shape[1], y * input_shape[0] / output_shape[0])
        kps_result[0, j, 2] = hm_j.max() / 255 

    # map back to original images
    for j in range(num_kps):
        kps_result[0, j, 0] = kps_result[0, j, 0] * ww_ori / input_shape[1] 
        kps_result[0, j, 1] = kps_result[0, j, 1] * hh_ori / input_shape[0] 

    return kps_result
