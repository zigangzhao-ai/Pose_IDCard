'''
code by zzg-2020-10-07
'''
# -*- coding: utf-8 -*-

import os
import random

#first create
os.makedirs('VOC2007/Annotations')
os.makedirs('VOC2007/ImageSets')
os.makedirs('VOC2007/ImageSets/Main')
os.makedirs('VOC2007/JPEGImages')

def _main():

    val_percent = 0.15   
    train_percent = 0.85

    xmlfilepath = "/workspace/zigangzhao/Pose_IDCard/scripts/all_data_0105/xml_train/"
    total_xml = os.listdir(xmlfilepath)

    num = len(total_xml)
    list = range(num)
    tv = int(num * val_percent)
    tr = int(num * train_percent)
 
    val = random.sample(list, tv)


    ftest = open('VOC2007/ImageSets/Main/test.txt', 'w')
    ftrain = open('VOC2007/ImageSets/Main/train.txt', 'w')

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in val:
            ftest.write(name)
        else:
            ftrain.write(name)

    ftest.close()
    ftrain.close()

if __name__ == '__main__':
    _main()