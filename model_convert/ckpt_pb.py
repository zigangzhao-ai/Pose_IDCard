'''
code by zzg 2021/01/21
ckpt_2_pb
'''
"""
此文件可以把ckpt模型转为pb模型
"""

import tensorflow as tf
#from create_tf_record import *
from tensorflow.python.framework import graph_util


def freeze_graph(input_checkpoint,output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB model
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #check the state of ckpt


    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # 直接用最后输出的节点，可以在tensorboard中查找到，tensorboard只能在linux中使用
    output_node_names = "tower_0/headnet/out/BiasAdd"
    #output_node_names = "tower_0/meshgrid/stack"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点


input_checkpoint='/workspace/zigangzhao/Pose_8_IDCard/output/model_dump/MPII/snapshot_180.ckpt'
# input_checkpoint="/workspace/zigangzhao/Pose_2020/TF-SimpleHumanPose/output/model_dump/MPII/snapshot_107.ckpt"
out_pb_path='frozen_model_mbv2.pb'
freeze_graph(input_checkpoint, out_pb_path)