import json
import numpy as np 
import math
import pdb
coco_gt = '/workspace/zigangzhao/Pose_IDCard/data/MPII/annnotations/test.json'
#coco_dt = '../output_fpn_nobbox/result/MPII/result.json'
coco_dt = '/workspace/zigangzhao/Pose_IDCard/output/result/MPII/result.json'

#6.96
#14.61
with open(coco_gt, 'r') as app:
	gt = json.load(app)
with open(coco_dt, 'r') as bpp:
	dt = json.load(bpp)

num_ignore = 0
num_positive = 0
distance = 0
new_list = []
cnt = 0

print(len(gt['annotations']))
print(len(dt))

img_id_list = []
for annogt in gt['annotations']:
	img_id = annogt['image_id']
	img_id_list.append(img_id)


for annogt in gt['annotations']:

	img_id = annogt['image_id']
	ketgt = annogt['keypoints']
	# print(ketgt)
	new_gt = []
	for i in range(4):
		new_gt.append([ketgt[i*3], ketgt[i*3+1], ketgt[i*3+2]])
	new_gt = np.array(new_gt)

	for annodt in dt:
		if annodt['image_id']==img_id:
			print(img_id)
			# cnt += 1
			# print(cnt)
			keydt = annodt['keypoints']
			new_dt = []
			for i in range(4):
				new_dt.append([keydt[i*3], keydt[i*3+1], keydt[i*3+2]])
			new_dt = np.array(new_dt)
			
			#pdb.set_trace()
			for (xd,yd,vd) in new_dt:
				dis_list = []
				if vd < 0.5:
					num_ignore += 1
					break
				else:
					num_positive += 1
					for (xg,yg,vg) in new_gt:
						dis = math.sqrt((xd-xg)*(xd-xg) + (yd-yg)*(yd-yg))
						dis_list.append(dis)
		
					# print(dis_list) 
					#min_dis = min(dis_list)
					mean_dis = sum(dis_list) / len(dis_list)
					# print('min:', min_dis)
					#new_list.append(min_dis)
					new_list.append(mean_dis)

print("the max num of new_list:", max(new_list))
print("the max num of new_list:", min(new_list))
print("length of new_list:", len(new_list))
print("num_ignore=", num_ignore)
print("num_positive=", num_positive)
print('mean_dis:', sum(new_list)/len(new_list))



