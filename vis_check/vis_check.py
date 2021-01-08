'''
code by zzg@2020/01/05
'''


from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import numpy as np

json_file = "check.json"
src_img_dir = "/workspace/zigangzhao/Pose_IDCard/scripts/all_data_0105/image_src/"
coco = COCO(json_file)
# print(coco.anns)

num_kps = 4
kps_names = ["l_up", "r_up", "r_down", "l_down"]


kps_symmetry = [(0, 1), (3, 2)]
kps_lines = [(0, 1), (1, 2), (2, 3), (3, 0)]

#Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]


for aid in coco.anns.keys():
    # print(aid)
    ann = coco.anns[aid]
    imgname = coco.imgs[ann['image_id']]['file_name']
    print(imgname)
    img = cv2.imread(src_img_dir + imgname)
    bbox = ann['bbox']
    print(bbox)
    [x1, y1, w, h] = bbox
    x2 = x1 + w
    y2 = y1 + h
    if imgname == 'face_1008.jpg':
        joints = ann['keypoints']
        print(joints)
        x11 = int(joints[0])
        y11 = int(joints[1])
        x22 = int(joints[3])
        y22 = int(joints[4])
        x33 = int(joints[6])
        y33 = int(joints[7])
        x44 = int(joints[9])
        y44 = int(joints[10])
        p1 = (x11, y11)
        p2 = (x22, y22)
        p3 = (x33, y33)
        p4 = (x44, y44)
        # cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 3)
        cv2.circle(img, p1, radius=5, color=colors[0], thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img, p2, radius=5, color=colors[1], thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img, p3, radius=5, color=colors[2], thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img, p4, radius=5, color=colors[3], thickness=-1, lineType=cv2.LINE_AA)
        # vis_keypoints(img, joints)
        plt.imshow(img)
        plt.show()
