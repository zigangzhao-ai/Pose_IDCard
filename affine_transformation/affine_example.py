'''
code by zzg@2020/01/08
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
# load a image
img = cv2.imread('0108.jpg')
rows, cols = img.shape[:2]

# original pts
pts_o = np.float32([[91, 271], [677, 192], [211, 760], [899, 628]]) # 这四个点为原始图片上数独的位置
pts_d = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]]) # 这是变换之后的图上四个点的位置

# get transform matrix
M = cv2.getPerspectiveTransform(pts_o, pts_d)
# apply transformation
dst = cv2.warpPerspective(img, M, (600, 600)) # 最后一参数是输出dst的尺寸。可以和原来图片尺寸不一致。按需求来确定

plt.imshow(img)
plt.show()
plt.imshow(dst)
plt.show()