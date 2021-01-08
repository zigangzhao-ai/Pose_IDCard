import os

from pycocotools.coco import COCO
from cocoeval import COCOeval
import json
import numpy as np
import pdb

#pdb.set_trace()
coco = COCO("/workspace/zigangzhao/Pose_IDCard/test_coco/dyh/test_dyh.json")
coco_dt = coco.loadRes("/workspace/zigangzhao/Pose_IDCard/test_coco/dyh/result_dyh.json")
#print(coco_dt)
coco_eval = COCOeval(coco, coco_dt, 'keypoints')
coco_eval.params.useSegm = None
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

info_str = []
for ind, name in enumerate(stats_names):
    info_str.append((name, coco_eval.stats[ind]))
