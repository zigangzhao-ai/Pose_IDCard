'''
code by zzg@2020/01/07
'''
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util


ckpt_path = "/workspace/zigangzhao/Pose_2020/TF-SimpleHumanPose/output/model_dump/MPII/snapshot_107.ckpt"
with tf.Session() as sess:
    for var_name, _ in tf.contrib.framework.list_variables(ckpt_path):
        print(var_name)
        var = tf.contrib.framework.load_variable(ckpt_path, var_name)
        #print(var_name, var.shape)
        # print(var)
