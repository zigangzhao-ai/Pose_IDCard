import os

from pycocotools.coco import COCO
from cocoeval import COCOeval
import json
import numpy as np
from collections import defaultdict
import pdb

def computeOks(imgId, catId, gts11, dts11):
    # dimention here should be Nxm
    #pdb.set_trace()
    gts = gts11[imgId, catId]
    dts = dts11[imgId, catId]
    # print(gts)
    # print(dts)
    inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
    dts = [dts[i] for i in inds]

    if len(gts) == 0 or len(dts) == 0:
        return []
    #ious = np.zeros((len(dts), len(gts)))
    ious = []
    sigmas = np.array([.25, .25, .25, .25])/10.0
    vars = (sigmas * 2)**2
    #vars = sigmas
    k = len(sigmas)
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt['keypoints'])
        xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
        k1 = np.count_nonzero(vg > 0)
        bb = gt['bbox']
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        for i, dt in enumerate(dts):
            d = np.array(dt['keypoints'])
            xd = d[0::3]; yd = d[1::3]
            if k1>0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2

            if k1 > 0:
                e=e[vg > 0]
            #ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
            ious.append(np.sum(np.exp(-e)) / e.shape[0])
    return ious

#pdb.set_trace()

coco_gt = COCO("zzg/test.json")
coco_dt = coco_gt.loadRes("/workspace/zigangzhao/Pose_IDCard/output/result/MPII/result.json")

imgIds = sorted(coco_gt.getImgIds())
catIds = sorted(coco_gt.getCatIds())

catIds = [0]  
'''
notion
"category_id": 0  we set it
it defaults to 1
'''
# print(catIds)
gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=imgIds))
dts = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=imgIds))


gts11 = defaultdict(list)       # gt for evaluation
dts11 = defaultdict(list)       # dt for evaluation

for gt in gts:
    gts11[gt['image_id'], gt['category_id']].append(gt)
for dt in dts:
    dts11[dt['image_id'], dt['category_id']].append(dt)

#pdb.set_trace()
ious = [computeOks(imgId, catId, gts11, dts11) \
                for imgId in imgIds
                for catId in catIds]
# print(ious)

ap50 = 0
ap75 = 0
for iou in ious:
	if iou[0] > 0.5:
		ap50 += 1
	if iou[0] > 0.75:
		ap75 += 1

print('ap50:', ap50/len(imgIds))
print('ap75:', ap75/len(imgIds))
