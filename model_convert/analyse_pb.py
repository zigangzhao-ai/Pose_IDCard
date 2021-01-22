"""
code by zzg
"""

import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
 
with tf.Session() as sess:
    with open('output_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # output = tf.import_graph_def(graph_def, return_elements=['cat']) 
        # print(graph_def, name='')
        # print(sess.run(output))
        tf.import_graph_def(graph_def, name='')
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        for tensor_name in tensor_name_list:
             print(tensor_name, '\n')