import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.summary import FileWriter
from tensorflow.python import pywrap_tensorflow
import pdb


def freeze_graph(input_checkpoint,output_graph): 
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "tower_0/out/BiasAdd"
    #output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        pdb.set_trace()
        
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
        
        #output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

## compute the flops by pb
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('  + Number of FLOPs: %.4fG' % (flops.total_float_ops / 1e9))
    #print('FLOPs: {};  Trainable params: {}'.format(flops.total_float_ops/1e9, params.total_parameters))

def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph



input_check = 'est_model/snapshot_140.ckpt'
output_check = 'test_model/snapshot_140.pb'
freeze_graph(input_check, output_check)
graph = load_pb('test_model/snapshot_140.pb')
print('stats after freezing')
stats_graph(graph)