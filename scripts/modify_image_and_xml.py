
'''
code by zzg 2020-05-30
'''
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# get annotation object bndbox location

try:
    import xml.etree.cElementTree as ET  
except ImportError:
    import xml.etree.ElementTree as ET

import os,sys
import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pdb

#the direction/path of Image,Label
src_img_dir = "all_data_0105/image_src"
src_xml_dir = "all_data_0105/xml_src"
dst_img_dir = "all_data_0105/image_all"
dst_xml_dir = "all_data_0105/xml_train"

if not os.path.exists(dst_img_dir):
    os.makedirs(dst_img_dir)

if not os.path.exists(dst_xml_dir):
    os.makedirs(dst_xml_dir)

img_Lists = glob.glob(src_img_dir + '/*.jpg')


img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))
    #print(img_basenames)

img_name = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_name.append(temp1)
# print(img_name)


cnt = 0
for img in img_name:

    im = cv2.imread(src_img_dir + '/' + img + '.jpg')
    # print(type(im))
    print(im.shape[::-1])
    channels, width, height = im.shape[::-1]  ##get w and h

    ##read the scr_xml
    AnotPath = src_xml_dir + '/' + img + '.xml'
    tree = ET.ElementTree(file=AnotPath)  
    root = tree.getroot()
    ObjectSet = root.findall('object')
    ObjBndBoxSet = []
    ObjBndBoxSet1 = {} 
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('keypoints')
        x1 = int(BndBox.find('x1').text) #- 1 
        y1 = int(BndBox.find('y1').text) #- 1
        x2 = int(BndBox.find('x2').text) #- 1
        y2 = int(BndBox.find('y2').text) #- 1
        x3 = int(BndBox.find('x3').text) #- 1 
        y3 = int(BndBox.find('y3').text) #- 1
        x4 = int(BndBox.find('x4').text) #- 1
        y4 = int(BndBox.find('y4').text) #- 1
        BndBoxLoc = [ObjName, x1,y1, x2,y2, x3,y3, x4,y4]
        # print(x1,y1,x2,y2)
        ObjBndBoxSet.append(BndBoxLoc) 
        print(ObjBndBoxSet)
    
    # save the crop-image in dst_crop
    cnt += 1
    cv2.imwrite(dst_img_dir + '/' + str(cnt) + '.jpg', im) #rename + '_' +

    # rewrite xml to dst_xml
    xml = open((dst_xml_dir + '/' + str(cnt)  + '.xml'), 'w')

    xml.write('<annotation>\n')
    xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
    xml.write('\t<filename>' + str(cnt)+ '.jpg' + '</filename>\n')
    xml.write('\t<source>\n')
    xml.write('\t\t<database>Unknown</database>\n')
    xml.write('\t</source>\n')
    xml.write('\t<size>\n')
    xml.write('\t\t<width>'+ str(width) + '</width>\n')
    xml.write('\t\t<height>'+ str(height) + '</height>\n')
    xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
    xml.write('\t</size>\n')
    xml.write('\t\t<segmented>0</segmented>\n')
      
    print("===========start rewrite bndbox==============")
    for x in ObjBndBoxSet:
        # print(x)
        [classname, x1,y1, x2,y2, x3,y3, x4,y4] = x   

        xml.write('\t<object>\n')
        xml.write('\t\t<name>'+ classname +'</name>\n')
        xml.write('\t\t<truncated>1</truncated>\n')
        xml.write('\t\t<difficult>0</difficult>\n')
        xml.write('\t\t<keypoints>\n')
        xml.write('\t\t\t<x1>' + str(x1) + '</x1>\n')
        xml.write('\t\t\t<y1>' + str(y1) + '</y1>\n')
        xml.write('\t\t\t<x1f>' + str(1.0) + '</x1f>\n')
        xml.write('\t\t\t<x2>' + str(x2) + '</x2>\n')
        xml.write('\t\t\t<y2>' + str(y2) + '</y2>\n')
        xml.write('\t\t\t<x2f>' + str(1.0) + '</x2f>\n')
        xml.write('\t\t\t<x3>' + str(x3) + '</x3>\n')
        xml.write('\t\t\t<y3>' + str(y3) + '</y3>\n')
        xml.write('\t\t\t<x3f>' + str(1.0) + '</x3f>\n')
        xml.write('\t\t\t<x4>' + str(x4) + '</x4>\n')
        xml.write('\t\t\t<y4>' + str(y4) + '</y4>\n')
        xml.write('\t\t\t<x4f>' + str(1.0) + '</x4f>\n')
        xml.write('\t\t</keypoints>\n')
        xml.write('\t</object>\n')       
            
    xml.write('</annotation>')

    print(cnt)

print("=======================finished!===================")
