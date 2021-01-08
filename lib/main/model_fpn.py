import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import json
import math
from functools import partial

from config import cfg
from tfflat.base import ModelDesc
import pdb

#from nets.basemodel import resnet50,resnet_arg_scope, resnet_v1, mobilenet_v2, mobilenet_v2_arg_scope
from nets.basemodel import mobilenet_v2_fpn, mobilenet_v2_arg_scope

class Model(ModelDesc):

     
    def head_net(self, blocks, is_training, trainable=True):
        
        normal_initializer = tf.truncated_normal_initializer(0, 0.01)
        msra_initializer = tf.contrib.layers.variance_scaling_initializer()
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        
        #aname = 'headnet_' + str(i)
        with tf.variable_scope('headnet') as scope:
            with slim.arg_scope(mobilenet_v2_arg_scope(cfg.weight_decay, is_training=is_training)):
            #with slim.arg_scope(mobilenet_v2_arg_scope(cfg.weight_decay, is_training=True, depth_multiplier=1.4, regularize_depthwise=False,dropout_keep_prob=1.0)):
                
                #out = slim.conv2d_transpose(blocks[-1], 256, [4, 4], stride=2,
                    #trainable=trainable, weights_initializer=normal_initializer,
                    #padding='SAME', activation_fn=tf.nn.relu,
                    #scope='up1')
                
                ##blocks[-1]
                '''
                out = slim.conv2d_transpose(blocks, 256, [4, 4], stride=2,
                    trainable=trainable, weights_initializer=normal_initializer,
                    padding='SAME', activation_fn=tf.nn.relu,scope='up1')
                
                  
                out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                    trainable=trainable, weights_initializer=normal_initializer,
                    padding='SAME', activation_fn=tf.nn.relu,
                    scope='up2')
                out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                    trainable=trainable, weights_initializer=normal_initializer,
                    padding='SAME', activation_fn=tf.nn.relu,
                    scope='up3')
                '''

                out = slim.conv2d(blocks, cfg.num_kps, [1, 1],
                        trainable=trainable, weights_initializer=msra_initializer,
                        padding='SAME', normalizer_fn=None, activation_fn=None,
                        scope='out')

        return out
        

   
    def render_gaussian_heatmap(self, coord, output_shape, sigma):
        
        x = [i for i in range(output_shape[1])]
        y = [i for i in range(output_shape[0])]
        xx,yy = tf.meshgrid(x,y)
        xx = tf.reshape(tf.to_float(xx), (1,*output_shape,1))
        yy = tf.reshape(tf.to_float(yy), (1,*output_shape,1))
              
        x = tf.floor(tf.reshape(coord[:,:,0],[-1,1,1,cfg.num_kps]) / cfg.input_shape[1] * output_shape[1] + 0.5)
        y = tf.floor(tf.reshape(coord[:,:,1],[-1,1,1,cfg.num_kps]) / cfg.input_shape[0] * output_shape[0] + 0.5)

        heatmap = tf.exp(-(((xx-x)/tf.to_float(sigma))**2)/tf.to_float(2) -(((yy-y)/tf.to_float(sigma))**2)/tf.to_float(2))

        return heatmap * 255.
   
    def make_network(self, is_train):
        #pdb.set_trace()
        if is_train:
            image = tf.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.input_shape, 3])
            target_coord = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps, 2])
            valid = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.num_kps])
            self.set_inputs(image, target_coord, valid)
        else:
            image = tf.placeholder(tf.float32, shape=[None, *cfg.input_shape, 3])
            self.set_inputs(image)

        #heatmap_outs = []
        backbone = eval(cfg.backbone)
        feature_maps = backbone(image, is_train)
        heatmap_outs = self.head_net(feature_maps[0], is_train)
        
        '''
        # output_node
        output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]
        print('out:', output_node_names)
        
        #pdb.set_trace()
        # flops
        g = tf.Graph()
        run_meta = tf.RunMetadata()
        with g.as_default():
            image11 = tf.placeholder(tf.float32, shape = [1, 256, 256, 3])
            feature_maps11 = backbone(image11, is_train)
            heatmap_outs11 = self.head_net(feature_maps11[0], is_train)

            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
            params = tf.profiler.profile(g, run_meta=run_meta, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()) 
            print('TF stats gives {} GFLOPs'.format(flops.total_float_ops/1e9))
            print('trainable params: {} M'.format(params.total_parameters/1e6))
        '''
        

        if is_train:
            gt_heatmap = tf.stop_gradient(self.render_gaussian_heatmap(target_coord, cfg.output_shape, cfg.sigma))
            valid_mask = tf.reshape(valid, [cfg.batch_size, 1, 1, cfg.num_kps])
            loss = tf.reduce_mean(tf.square(heatmap_outs - gt_heatmap) * valid_mask)
            self.add_tower_summary('loss', loss)
            self.set_loss(loss)
        else:
            self.set_outputs(heatmap_outs)

