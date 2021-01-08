from pycocotools.coco import COCO

test_annot_path = "/workspace/zigangzhao/Pose_IDCard/data/MPII/annnotations/test.json"
coco = COCO(test_annot_path)
    
for aid in coco.anns.keys():
    # print(aid)
    # print(coco.anns)
    ann = coco.anns[aid]
    # print(ann)
    # print(ann['image_id'])
    # print(coco.imgs)
    print(coco.imgs[1])
    imgname = coco.imgs[ann['image_id']]['file_name']
    # print(imgname)
    joints = ann['keypoints']
