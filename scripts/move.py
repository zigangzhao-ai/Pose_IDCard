'''
code by zzg -2020-10-07
'''
##split total image dataset for train and test
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os,sys
import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import pdb

#the direction/path of Image,Label
# src_img_dir = "/workspace/zigangzhao/Pose_IDCard/scripts/all_data_0105/image_train/"
src_xml_dir = "/workspace/zigangzhao/Pose_IDCard/scripts/all_data_0105/xml_train/"
test_txt = "/workspace/zigangzhao/Pose_IDCard/scripts/VOC2007/ImageSets/Main/test.txt"

# dst_img_dir = "/workspace/zigangzhao/Pose_IDCard/scripts/all_data_0105/image_test/"
dst_xml_dir = "/workspace/zigangzhao/Pose_IDCard/scripts/all_data_0105/xml_test/"

# if not os.path.exists(dst_img_dir):
#     os.makedirs(dst_img_dir)

if not os.path.exists(dst_xml_dir):
    os.makedirs(dst_xml_dir)

img_Lists = glob.glob(src_xml_dir + '/*.xml')
# print(img_Lists)
# rename = src_img_dir.split('/')[-1]
# print(rename)

img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))
    #print(img_basenames)

img_name = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_name.append(temp1)
print(len(img_name))

val_list = []
with open(test_txt, "r") as f:
    lines = f.readlines()
    for x in lines:
        x = x.replace('\n', '')     
        val_list.append(str(x))
print(val_list)

cnt = 0
for img in img_name:
    #print(img)
    if img in val_list:
        print("====")
        cnt += 1
        print(cnt)
        # img_path = src_img_dir + '/' + img + '.jpg'
        # shutil.move(img_path, dst_img_dir)

        xml_path = src_xml_dir + '/' + img + '.xml'
        shutil.move(xml_path, dst_xml_dir)

print("total move image is {}".format(cnt))
print("finished move !!")