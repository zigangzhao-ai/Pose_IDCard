'''
code by zzg-2021/01/27
'''
# input_names = ["tower_0/Placeholder"]/["tower_0/MobilenetV2/input"]
# out_names = ["tower_0/headnet/out/BiasAdd"]
# input_tensor = [1, 256, 192,3]
# output_tensor = [1, 8, 3]
# [width, height] = [192, 256]

import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import math

from matplotlib import pyplot as plt
from utils import vis_keypoints, convert 
from inference_pb import inference_pb
from PIL import Image
import pdb
 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #"0, 1"
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 

##config
input_shape = (256, 192)
[width, height] = [192, 256]
flip_test = True
output_shape = (input_shape[0]//4, input_shape[1]//4)
num_kps = 8

pixel_means = np.array([[[123.68, 116.78, 103.94]]])
def normalize_input(img):
    return img - pixel_means


if __name__=="__main__":
    # Path to frozen detection graph. This is the actual model that is used for the keypoints detection.
    PATH_TO_PB = "/workspace/zigangzhao/Pose_8_IDCard/model_convert/pb/frozen_model_mbv2.pb"

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_PB, 'rb') as fid:
            serialized_graph = fid.read()
            #print(serialized_graph)
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    #If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    '''
    PATH_TO_TEST_IMAGES_DIR = 'test_vis'
    total_name = os.listdir(PATH_TO_TEST_IMAGES_DIR) 
    a = []
    for i in range(0, len(total_name)):
        name = total_name[i][:-4]
        #print(name)
        a.append(name) 
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(x)) for x in a]
    '''
    ##test a single image
    image_path = "/workspace/zigangzhao/Pose_8_IDCard/tool/test_image/030202.jpg"
    # input_type = "IDCard_Reverse"   #"IDCard_front"
    input_type = "IDCard_front"
    name = image_path.split('/')[-1].split('.')[0]
    image_path_save = "test_vis/" + name
    if not os.path.exists(image_path_save):
        os.mkdir(image_path_save)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:       
            sess = tf.Session(graph=detection_graph, config=config)
            
            # for image_path in TEST_IMAGE_PATHS:
            ori_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            hh_ori, ww_ori, _ = ori_image.shape

            val_image = cv2.resize(ori_image, (width, height)).astype(np.float32)            
            #print(type(val_image))
            image_np = normalize_input(val_image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)        
            #input.
            image_tensor = detection_graph.get_tensor_by_name('tower_0/Placeholder:0')
            #print(image_tensor)        
            # outputs of the image.
            outputs = detection_graph.get_tensor_by_name('tower_0/headnet/out/BiasAdd:0')
            # Actual detection.
            output = sess.run([outputs], feed_dict={image_tensor: image_np_expanded})

            output = np.array(output)[0, ...]
            # output = np.expand_dims(np.squeeze(np.array(output)), axis=0) 
            print(output.shape)
            kps_result = convert(output, hh_ori, ww_ori)
            print(kps_result)
            # print(kps_result.shape)
            # print(len(kps_result))
            for i in range(len(kps_result)):
                tmpkps = np.zeros((3, num_kps))
                tmpkps[:2, :] = kps_result[i, :, :2].transpose(1,0)
                tmpkps[2, :] = kps_result[i, :, 2]
                tmpimg, cnt = vis_keypoints(ori_image, tmpkps)
                kps = tmpkps
                print(cnt)

                if input_type == "IDCard_front":

                    x1, y1 = int(kps[0, 0]), int(kps[1, 0])
                    x2, y2 = int(kps[0, 1]), int(kps[1, 1])
                    x3, y3 = int(kps[0, 2]), int(kps[1, 2])
                    x4, y4 = int(kps[0, 3]), int(kps[1, 3])
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
                    x_o = (xmin + xmax) // 2
                    y_o = (ymin + ymax) // 2
                    
                    if cnt == 3:
                        p1 = [xmin, ymin]
                        p2 = [xmin+w, ymin]
                        p3 = [xmax, ymax]
                        p4 = [xmin, ymin+h]
                        pts_o = np.float32([p1, p2, p3, p4]) # ori

                        if w > h:
                            pts_d = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]) # transfer  
                            # get transform matrix
                            M = cv2.getPerspectiveTransform(pts_o, pts_d)
                            # apply transformation
                            dst = cv2.warpPerspective(ori_image, M, (w, h))
                            dst = cv2.resize(dst, dsize=(480, 270),interpolation=cv2.INTER_CUBIC)
                            _, cnt_out = inference_pb(dst)
                        else:
                            pts_d = np.float32([[[0, 0], [h, 0], [h, w], [0, w]]]) # transfer  
                            # get transform matrix
                            M = cv2.getPerspectiveTransform(pts_o, pts_d)
                            # apply transformation
                            dst = cv2.warpPerspective(ori_image, M, (h, w))
                            dst = cv2.resize(dst, dsize=(480, 270),interpolation=cv2.INTER_CUBIC) 
                            _, cnt_out = inference_pb(dst)

                        plt.figure()
                        plt.subplot(1,2,1)
                        plt.imshow(tmpimg)
                        plt.subplot(1,2,2)
                        plt.imshow(dst)
                        plt.show()
                        cv2.imwrite(image_path_save + '/{}.jpg'.format('001'), dst)
                
                    if cnt == 4:
        
                        a = [p1, p2, p3, p4]
                        b = list(set(map(tuple, a)))
                        a1, b1 = [x1, x2, x3, x4], [y1, y2, y3, y4]
                        a11, b11 = set(a1), set(b1)
                        if (len(a11)==3 or len(b11)==3) and x2 > x1 and y4 > y1:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [xmax, ymax]
                            p4 = [xmin, ymax]

                        if len(b) == 4 and y2 > y1 and x3 < x2 and y3 < y4 and x3 < x_o:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [x4, y4]
                            p4 = [x3, y3]

                        if len(b) == 4 and y2 < y1 and x3 > x2 and y3 > y4 and y4 < y_o:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [x4, y4]
                            p4 = [x3, y3]

                        pts_o = np.float32([p1, p2, p3, p4]) # ori

                        if w > h:
                            pts_d = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]) # transfer  
                            # get transform matrix
                            M = cv2.getPerspectiveTransform(pts_o, pts_d)
                            # apply transformation
                            dst = cv2.warpPerspective(ori_image, M, (w, h)) 
                            dst = cv2.resize(dst, dsize=(480, 270),interpolation=cv2.INTER_CUBIC)
                            _, cnt_out = inference_pb(dst)
                        else:
                            pts_d = np.float32([[[0, 0], [h, 0], [h, w], [0, w]]]) # transfer  
                            # get transform matrix
                            M = cv2.getPerspectiveTransform(pts_o, pts_d)
                            # apply transformation
                            dst = cv2.warpPerspective(ori_image, M, (h, w)) 
                            dst = cv2.resize(dst, dsize=(480, 270),interpolation=cv2.INTER_CUBIC)
                            _, cnt_out = inference_pb(dst)

                        plt.figure()
                        plt.subplot(1,2,1)
                        plt.imshow(tmpimg)
                        plt.subplot(1,2,2)
                        plt.imshow(dst)
                        plt.show()
                        cv2.imwrite(image_path_save + '/{}.jpg'.format('001'), dst)
                    
                    if cnt >= 5 and cnt <= 8:
            
                        x1, y1 = int(kps[0, 0]), int(kps[1, 0])
                        x2, y2 = int(kps[0, 1]), int(kps[1, 1])
                        x3, y3 = int(kps[0, 2]), int(kps[1, 2])
                        x4, y4 = int(kps[0, 3]), int(kps[1, 3])

                        x11, y11 = int(kps[0, 4]), int(kps[1, 4])
                        x22, y22 = int(kps[0, 5]), int(kps[1, 5])
                        x33, y33 = int(kps[0, 6]), int(kps[1, 6])
                        x44, y44 = int(kps[0, 7]), int(kps[1, 7])

                        p1, p11 = [x1, y1], [x11, y11]
                        p2, p22 = [x2, y2], [x22, y22]
                        p3, p33 = [x3, y3], [x33, y33]
                        p4, p44 = [x4, y4], [x44, y44]

                        xmin, xmin1 = min(x1, x2, x3, x4), min(x11, x22, x33, x44)
                        ymin, ymin1 = min(y1, y2, y3, y4), min(y11, y22, y33, y44)
                        xmax, xmax1 = max(x1, x2, x3, x4), max(x11, x22, x33, x44)
                        ymax, ymax1 = max(y1, y2, y3, y4), max(y11, y22, y33, y44)
                        w, w1 = xmax - xmin, xmax1 - xmin1
                        h, h1 = ymax - ymin, ymax1 - ymin1
                        x_o = (xmin + xmax) // 2
                        y_o = (ymin + ymax) // 2

                        a = [p1, p2, p3, p4]
                        b = list(set(map(tuple, a)))
                        a1, b1 = [x1, x2, x3, x4], [y1, y2, y3, y4]
                        a11, b11 = set(a1), set(b1)
                        if (len(a11) == 3 or len(b11)==3) and x2 > x1 and y4 > y1:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [xmax, ymax]
                            p4 = [xmin, ymax]

                        if len(b) == 4 and x3 < x_o and w > h:
                            p1 = [xmin, ymin]
                            p2 = [xmin+w, ymin]
                            p3 = [xmax, ymax]
                            p4 = [xmin, ymin+h]

                        if len(b) == 4 and y2 > y1 and x3 < x2 and y3 < y4 and x3 < x_o and y3 > y_o:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [x4, y4]
                            p4 = [x3, y3]

                        if len(b) == 4 and y2 < y1 and x3 > x2 and y3 > y4 and y4 < y_o:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [x4, y4]
                            p4 = [x3, y3]
                        
                        if len(b) == 4 and y2 < y1 and x3 > x2 and y3 < y4 and x1 < x4 and w > h:
                            p1 = [x2, y2]
                            p2 = [x3, y3]
                            p3 = [x4, y4]
                            p4 = [x1, y1]

                        if len(b) == 4 and w < h and x1 > x_o and x2 > x_o and y1 < y_o and y2 > y_o and y3 < y_o:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [x4, y4]
                            p4 = [x3, y3]
                            w, h = h, w

                        c, d = [x11, x22, x33, x44], [y11, y22, y33, y44]
                        e, f = set(c), set(d)

                        if (len(e) == 3 or len(f)== 3) and x22 > x11 and y44 > y11:
                            p11 = [x11, y11]
                            p22 = [x22, y22]
                            p33 = [xmax1, ymax1]
                            p44 = [xmin1, ymax1]

                        pts_o = np.float32([p1, p2, p3, p4]) # ori
                        pts_o1 = np.float32([p11, p22, p33, p44])

                        if w > h:
                            pts_d = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]) # transfer
                            pts_d1 = np.float32([[[0, 0], [w1, 0], [w1, h1], [0, h1]]]) # transfer    
                            # get transform matrix
                            M = cv2.getPerspectiveTransform(pts_o, pts_d)
                            M1 = cv2.getPerspectiveTransform(pts_o1, pts_d1)
                            # apply transformation
                            dst = cv2.warpPerspective(ori_image, M, (w, h))
                            dst = cv2.resize(dst, dsize=(480, 270), interpolation=cv2.INTER_CUBIC)
                            dst1 = cv2.warpPerspective(ori_image, M1, (w1, h1))
                            dst1 = cv2.resize(dst1, dsize=(200, 280), interpolation=cv2.INTER_CUBIC)
                            _, cnt_out = inference_pb(dst)
                            _, cnt_out1 = inference_pb(dst1)
                        else:   
                            pts_d = np.float32([[[0, 0], [h, 0], [h, w], [0, w]]]) # transfer
                            pts_d1 = np.float32([[[0, 0], [h1, 0], [h1, w1], [0, w1]]]) # transfer    
                            # get transform matrix
                            M = cv2.getPerspectiveTransform(pts_o, pts_d)
                            M1 = cv2.getPerspectiveTransform(pts_o1, pts_d1)
                            # apply transformation
                            dst = cv2.warpPerspective(ori_image, M, (w, h))
                            dst = cv2.resize(dst, dsize=(480, 270), interpolation=cv2.INTER_CUBIC)
                            dst1 = cv2.warpPerspective(ori_image, M1, (w1, h1))
                            dst1 = cv2.resize(dst1, dsize=(200, 280), interpolation=cv2.INTER_CUBIC)
                            _, cnt_out = inference_pb(dst)
                            _, cnt_out1 = inference_pb(dst1)
                        
                        plt.figure()
                        plt.subplot(1,3,1)
                        plt.imshow(tmpimg[:,:,[2,1,0]])
                        plt.subplot(1,3,2)
                        plt.imshow(dst[:,:,[2,1,0]])
                        plt.subplot(1,3,3)
                        plt.imshow(dst1[:,:,[2,1,0]])
                        plt.show()
                        cv2.imwrite(image_path_save + '/{}.jpg'.format('001'), dst)
                        cv2.imwrite(image_path_save + '/{}.jpg'.format('002'), dst1)
                        
                    else:
                        cnt_out = None
                        
                    print(cnt, cnt_out)
                    
                    if cnt_out == None:
                        print("failed")
                    if cnt <= 4 and cnt_out:
                        if cnt_out >= 1:
                            print(cnt_out)
                            print("successful")
                        else: 
                            print("failed")
                    if cnt > 4 and cnt_out and cnt_out1:
                        if cnt_out >= 1 and cnt_out1 >= 1:
                            print(cnt_out, cnt_out1)
                            print("successful")
                        else: 
                            print("failed")

                if input_type == "IDCard_Reverse" or input_type == "DriverCard":

                    x1, y1 = int(kps[0, 0]), int(kps[1, 0])
                    x2, y2 = int(kps[0, 1]), int(kps[1, 1])
                    x3, y3 = int(kps[0, 2]), int(kps[1, 2])
                    x4, y4 = int(kps[0, 3]), int(kps[1, 3])
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

                    x_o = (xmin + xmax) // 2
                    y_o = (ymin + ymax) // 2

                    if cnt <= 2:
                        cnt_out = None

                    if cnt == 3:                       

                        if y2 > y1 and x3 < x2 and y3 < y4:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [x4, y4]
                            p4 = [x3, y3]
                        else:
                            xmin = min(x1, x2, x3, x4)
                            ymin = min(y1, y2, y3, y4)
                            xmax = max(x1, x2, x3, x4)
                            ymax = max(y1, y2, y3, y4)
                            w = xmax - xmin
                            h = ymax - ymin

                            p1 = [xmin, ymin]
                            p2 = [xmin+w, ymin]
                            p3 = [xmax, ymax]
                            p4 = [xmin, ymin+h]

                        pts_o = np.float32([p1, p2, p3, p4]) # ori

                        if w > h:
                            pts_d = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]) # transfer  
                            # get transform matrix
                            M = cv2.getPerspectiveTransform(pts_o, pts_d)
                            # apply transformation
                            dst = cv2.warpPerspective(ori_image, M, (w, h))
                            dst = cv2.resize(dst, dsize=(480, 270),interpolation=cv2.INTER_CUBIC)
                            _, cnt_out = inference_pb(dst)
                        else:
                            pts_d = np.float32([[[0, 0], [h, 0], [h, w], [0, w]]]) # transfer  
                            # get transform matrix
                            M = cv2.getPerspectiveTransform(pts_o, pts_d)
                            # apply transformation
                            dst = cv2.warpPerspective(ori_image, M, (h, w))
                            dst = cv2.resize(dst, dsize=(480, 270),interpolation=cv2.INTER_CUBIC)
                            _, cnt_out = inference_pb(dst) 

                        plt.figure()
                        plt.subplot(1,2,1)
                        plt.imshow(tmpimg)
                        plt.subplot(1,2,2)
                        plt.imshow(dst)
                        plt.show()
                        cv2.imwrite(image_path_save + '/{}.jpg'.format('001'), dst)
                
                    if cnt >= 4:

                        a = [p1, p2, p3, p4]
                        b = list(set(map(tuple, a)))
                        a1, b1 = [x1, x2, x3, x4], [y1, y2, y3, y4]
                        a11, b11 = set(a1), set(b1)

                        if (len(a11)==3 or len(b11)==3) and x2 > x1 and y4 > y1:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [xmax, ymax]
                            p4 = [xmin, ymax]

                        if len(b) == 4 and y2 > y1 and x3 < x2 and y3 < y4 and x3 < x_o:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [x4, y4]
                            p4 = [x3, y3]

                        if len(b) == 4 and y2 < y1 and x3 > x2 and y3 > y4 and y4 < y_o:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [x4, y4]
                            p4 = [x3, y3]
                            
                        if len(b) == 4 and w < h and x1 > x_o and x2 > x_o and y1 < y_o and y2 > y_o and y3 < y_o:
                            p1 = [x1, y1]
                            p2 = [x2, y2]
                            p3 = [x4, y4]
                            p4 = [x3, y3]
                            w, h = h, w

                        pts_o = np.float32([p1, p2, p3, p4]) # ori

                        if w > h:
                            pts_d = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]) # transfer  
                            # get transform matrix
                            M = cv2.getPerspectiveTransform(pts_o, pts_d)
                            # apply transformation
                            dst = cv2.warpPerspective(ori_image, M, (w, h)) 
                            dst = cv2.resize(dst, dsize=(480, 270),interpolation=cv2.INTER_CUBIC)
                            _, cnt_out = inference_pb(dst)
                        else:
                            pts_d = np.float32([[[0, 0], [h, 0], [h, w], [0, w]]]) # transfer  
                            # get transform matrix
                            M = cv2.getPerspectiveTransform(pts_o, pts_d)
                            # apply transformation
                            dst = cv2.warpPerspective(ori_image, M, (h, w)) 
                            dst = cv2.resize(dst, dsize=(480, 270),interpolation=cv2.INTER_CUBIC)
                            _, cnt_out = inference_pb(dst)

                        plt.figure()
                        plt.subplot(1,2,1)
                        plt.imshow(tmpimg)
                        plt.subplot(1,2,2)
                        plt.imshow(dst)
                        plt.show()
                        cv2.imwrite(image_path_save + '/{}.jpg'.format('001'), dst)
                                    
                    if cnt_out and cnt_out >= 1:
                        print(cnt_out)
                        print("successful")
                    else:
                        print("failed")
            cv2.imwrite(image_path_save + '/{}.jpg'.format(name), tmpimg)

    print("finished!!")
