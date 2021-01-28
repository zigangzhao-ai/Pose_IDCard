
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
src_img_dir = "/workspace/zigangzhao/0116_Card/all/image_src"
src_xml_dir = "/workspace/zigangzhao/0116_Card/all/xml_src/"
dst_img_dir = "/workspace/zigangzhao/0116_Card/all/image_all"
dst_xml_dir = "/workspace/zigangzhao/0116_Card/all/xml_train"

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
        x1f = float(BndBox.find('x1f').text)
        x2 = int(BndBox.find('x2').text) #- 1
        y2 = int(BndBox.find('y2').text) #- 1
        x2f = float(BndBox.find('x2f').text)
        x3 = int(BndBox.find('x3').text) #- 1 
        y3 = int(BndBox.find('y3').text) #- 1
        x3f = float(BndBox.find('x3f').text)
        x4 = int(BndBox.find('x4').text) #- 1
        y4 = int(BndBox.find('y4').text) #- 1
        x4f = float(BndBox.find('x4f').text)
        
        x5 = int(BndBox.find('x5').text) #- 1 
        y5 = int(BndBox.find('y5').text) #- 1
        x5f = float(BndBox.find('x5f').text)
        x6 = int(BndBox.find('x6').text) #- 1
        y6 = int(BndBox.find('y6').text) #- 1
        x6f = float(BndBox.find('x6f').text)
        x7 = int(BndBox.find('x7').text) #- 1 
        y7 = int(BndBox.find('y7').text) #- 1
        x7f = float(BndBox.find('x7f').text)
        x8 = int(BndBox.find('x8').text) #- 1
        y8 = int(BndBox.find('y8').text) #- 1
        x8f = float(BndBox.find('x8f').text)
        BndBoxLoc = [ObjName, x1,y1,x1f, x2,y2,x2f, x3,y3,x3f, x4,y4,x4f, x5,y5,x5f, x6,y6,x6f, x7,y7,x7f, x8,y8,x8f]
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
        [classname, x1,y1,x1f, x2,y2,x2f, x3,y3,x3f, x4,y4,x4f, x5,y5,x5f, x6,y6,x6f, x7,y7,x7f, x8,y8,x8f] = x   

        xml.write('\t<object>\n')
        xml.write('\t\t<name>'+ classname +'</name>\n')
        xml.write('\t\t<truncated>1</truncated>\n')
        xml.write('\t\t<difficult>0</difficult>\n')
        xml.write('\t\t<keypoints>\n')
        xml.write('\t\t\t<x1>' + str(x1) + '</x1>\n')
        xml.write('\t\t\t<y1>' + str(y1) + '</y1>\n')
        xml.write('\t\t\t<x1f>' + str(x1f) + '</x1f>\n')
        xml.write('\t\t\t<x2>' + str(x2) + '</x2>\n')
        xml.write('\t\t\t<y2>' + str(y2) + '</y2>\n')
        xml.write('\t\t\t<x2f>' + str(x2f) + '</x2f>\n')
        xml.write('\t\t\t<x3>' + str(x3) + '</x3>\n')
        xml.write('\t\t\t<y3>' + str(y3) + '</y3>\n')
        xml.write('\t\t\t<x3f>' + str(x3f) + '</x3f>\n')
        xml.write('\t\t\t<x4>' + str(x4) + '</x4>\n')
        xml.write('\t\t\t<y4>' + str(y4) + '</y4>\n')
        xml.write('\t\t\t<x4f>' + str(x4f) + '</x4f>\n')
        
        xml.write('\t\t\t<x5>' + str(x5) + '</x5>\n')
        xml.write('\t\t\t<y5>' + str(y5) + '</y5>\n')
        xml.write('\t\t\t<x5f>' + str(x5f) + '</x5f>\n')
        xml.write('\t\t\t<x6>' + str(x6) + '</x6>\n')
        xml.write('\t\t\t<y6>' + str(y6) + '</y6>\n')
        xml.write('\t\t\t<x6f>' + str(x6f) + '</x6f>\n')
        xml.write('\t\t\t<x7>' + str(x7) + '</x7>\n')
        xml.write('\t\t\t<y7>' + str(y7) + '</y7>\n')
        xml.write('\t\t\t<x7f>' + str(x7f) + '</x7f>\n')
        xml.write('\t\t\t<x8>' + str(x8) + '</x8>\n')
        xml.write('\t\t\t<y8>' + str(y8) + '</y8>\n')
        xml.write('\t\t\t<x8f>' + str(x8f) + '</x8f>\n')
        xml.write('\t\t</keypoints>\n')
        xml.write('\t</object>\n')       
            
    xml.write('</annotation>')

    print(cnt)

print("=======================finished!===================")
