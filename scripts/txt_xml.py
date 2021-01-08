'''
code by zzg@2021/01/05
'''
import os
import glob
from PIL import Image
import pdb
import cv2

# the direction/path of Image,Label
src_img_dir = "/workspace/zigangzhao/Pose_IDCard/scripts/data_20210104/image_and_annatation/Driver_front/"
src_txt_dir = "/workspace/zigangzhao/Pose_IDCard/scripts/data_20210104/image_and_annatation/Driver_front_txt/"
src_xml_dir = "/workspace/zigangzhao/Pose_IDCard/scripts/data_20210104/image_and_annatation/Driver_front_xml/"

if not os.path.exists(src_xml_dir):
    os.makedirs(src_xml_dir)

img_Lists = glob.glob(src_img_dir + '/*.jpg')
#print(img_Lists)

img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))
    #print(img_basenames)

img_name = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_name.append(temp1)
    #print(img_name)

for img in img_name:

    im = cv2.imread(src_img_dir + '/' + img + '.jpg')
    print(im.shape[::-1])
    channels, width, height = im.shape[::-1]  ##get w and h

    xml_file = open((src_xml_dir + '/' + str(img) + '.xml'), 'w')

    xml_file.write('<annotation>\n')
    xml_file.write('\t<folder>' + 'VOC2007' + '</folder>\n')
    xml_file.write('\t<filename>' + str(img) + '.jpg' + '</filename>\n')
    xml_file.write('\t<source>\n')
    xml_file.write('\t\t<database>Unknown</database>\n')
    xml_file.write('\t</source>\n')
    xml_file.write('\t<size>\n')
    xml_file.write('\t\t<width>'+ str(width) + '</width>\n')
    xml_file.write('\t\t<height>'+ str(height) + '</height>\n')
    xml_file.write('\t\t<depth>' + str(channels) + '</depth>\n')
    xml_file.write('\t</size>\n')
    xml_file.write('\t\t<segmented>0</segmented>\n')
    
    gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()

    for x in gt:
        spt = x.split(' ')
        print(spt)
        x1 = spt[0]
        y1 = spt[1]
        x2 = spt[2]
        y2 = spt[3]
        x3 = spt[4]
        y3 = spt[5]
        x4 = spt[6]
        y4 = spt[7]

        if x1 < x2 and x3 < x4:
            print("maybe need correct")
            x3, y3 = spt[6], spt[7] 
            x4, y4 = spt[4], spt[5]

        xml_file.write('\t<object>\n')
        xml_file.write('\t\t<name>'+ 'card' +'</name>\n')
        xml_file.write('\t\t<pose>Unspecified</pose>\n')
        xml_file.write('\t\t<truncated>1</truncated>\n')
        xml_file.write('\t\t<difficult>0</difficult>\n')
        xml_file.write('\t\t<keypoints>\n')
        xml_file.write('\t\t\t<x1>' + str(x1) + '</x1>\n')
        xml_file.write('\t\t\t<y1>' + str(y1) + '</y1>\n')
        xml_file.write('\t\t\t<x1f>' + str(1.0) + '</x1f>\n')
        xml_file.write('\t\t\t<x2>' + str(x2) + '</x2>\n')
        xml_file.write('\t\t\t<y2>' + str(y2) + '</y2>\n')
        xml_file.write('\t\t\t<x2f>' + str(1.0) + '</x2f>\n')
        xml_file.write('\t\t\t<x3>' + str(x3) + '</x3>\n')
        xml_file.write('\t\t\t<y3>' + str(y3) + '</y3>\n')
        xml_file.write('\t\t\t<x3f>' + str(1.0) + '</x3f>\n')
        xml_file.write('\t\t\t<x4>' + str(x4) + '</x4>\n')
        xml_file.write('\t\t\t<y4>' + str(y4) + '</y4>\n')
        xml_file.write('\t\t\t<x4f>' + str(1.0) + '</x4f>\n')
        xml_file.write('\t\t</keypoints>\n')
        xml_file.write('\t</object>\n')         
                
    xml_file.write('</annotation>')
    print("finished!")
