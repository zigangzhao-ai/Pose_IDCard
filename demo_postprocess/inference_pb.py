import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from matplotlib import pyplot as plt
from utils import vis_keypoints, convert 

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

def inference_pb(image):
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

    ##test a single image
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:       
            sess = tf.Session(graph=detection_graph, config=config)         
            # for image_path in TEST_IMAGE_PATHS:
            ori_image = image
            hh_ori, ww_ori, _ = ori_image.shape

            val_image = cv2.resize(ori_image, (width, height)).astype(np.float32)            
            #print(type(val_image))
            image_np = normalize_input(val_image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)         
            #input.
            image_tensor = detection_graph.get_tensor_by_name('tower_0/Placeholder:0')   
            # outputs of the image.
            outputs = detection_graph.get_tensor_by_name('tower_0/headnet/out/BiasAdd:0')
            # Actual detection.
            output = sess.run([outputs], feed_dict={image_tensor: image_np_expanded})
            output = np.array(output)[0, ...]
            # output = np.expand_dims(np.squeeze(np.array(output)), axis=0)   
            kps_result = convert(output, hh_ori, ww_ori)
            for i in range(len(kps_result)):
                tmpkps = np.zeros((3, num_kps))
                tmpkps[:2, :] = kps_result[i, :, :2].transpose(1,0)
                tmpkps[2, :] = kps_result[i, :, 2]
                tmpimg, cnt = vis_keypoints(ori_image, tmpkps, kp_thresh=0.3, alpha=1)

            return tmpimg, cnt
    

if __name__=="__main__":
    image_path = "/workspace/zigangzhao/Pose_8_IDCard/test_vis/030101/001.jpg"
    # image_path = "test_image/9.jpg"
    image_name = image_path.split('/')[-1].split('.')[0]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    tmpimg, result = inference_pb(image)
    # print(result)
    plt.imshow(tmpimg)
    plt.show()

    if result >= 1:
        print("successful")
    else:
        print("failed")