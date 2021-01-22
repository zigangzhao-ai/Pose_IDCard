'''
code by zzg@2020/01/08
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
# load a image
img = cv2.imread("/workspace/zigangzhao/Pose_IDCard/affine_transformation/283.jpg")
rows, cols = img.shape[:2]


kps =[[6.20833333, 284.03125, 291.79166667, 0.0 ],
      [190.0,      182.5,     370.0,       370.0    ],
      [0.79864975, 0.74664761, 0.76240713, 0.7505422 ]]

# print(kps)

x1, y1 = int(kps[0][0]), int(kps[1][0])
x2, y2 = int(kps[0][1]), int(kps[1][1])
x3, y3 = int(kps[0][2]), int(kps[1][2])
x4, y4 = int(kps[0][3]), int(kps[1][3])
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
pts_d = np.float32([[0, 0], [w, 0], [w, h], [0, h]]) # transfer
# get transform matrix
M = cv2.getPerspectiveTransform(pts_o, pts_d)
print(M)
# apply transformation
dst = cv2.warpPerspective(img, M, (w, h)) 
# original pts
# pts_o = np.float32([[91, 271], [677, 192], [211, 760], [899, 628]]) # 这四个点为原始图片上数独的位置
# pts_d = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]]) # 这是变换之后的图上四个点的位置

# # get transform matrix
# M = cv2.getPerspectiveTransform(pts_o, pts_d)
# # apply transformation
# dst = cv2.warpPerspective(img, M, (600, 600)) # 最后一参数是输出dst的尺寸。可以和原来图片尺寸不一致。按需求来确定


plt.figure()
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(dst)
plt.show()