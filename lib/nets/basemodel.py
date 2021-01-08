import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import slim as contrib_slim
from . import resnet_v1, resnet_utils
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import regularizers, initializers, layers
import sys
sys.path.append("/workspace/zigangzhao/Pose_IDCard/")
from main.config import cfg
import numpy as np
from . import mobilenet_v2 as mobv
from . import shape_utils
import functools
import collections

def resnet_arg_scope(bn_is_training,
                     bn_trainable,
                     trainable=True,
                     weight_decay=cfg.weight_decay,
                     weight_init = initializers.variance_scaling_initializer(),
                     batch_norm_decay=0.99,
                     batch_norm_epsilon=1e-9,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': bn_is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': bn_trainable,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=weight_init,
            trainable=trainable,
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

def resnet50(inp, bn_is_training, bn_trainable):
    bottleneck = resnet_v1.bottleneck
    blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 1)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 2)] + [(512, 128, 1)] * 3),
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 2)] + [(1024, 256, 1)] * 5),
        resnet_utils.Block('block4', bottleneck,
                           [(2048, 512, 2)] + [(2048, 512, 1)] * 2)
    ]   
    
    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):

        with tf.variable_scope('resnet_v1_50', 'resnet_v1_50'):
            net = resnet_utils.conv2d_same(
                    tf.concat(inp,axis=3), 64, 7, stride=2, scope='conv1')
            
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')
        net, _ = resnet_v1.resnet_v1(                                  # trainable ?????
            net, blocks[0:1],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')
    
    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net2, _ = resnet_v1.resnet_v1(
            net, blocks[1:2],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')

    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net3, _ = resnet_v1.resnet_v1(
            net2, blocks[2:3],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')

    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net4, _ = resnet_v1.resnet_v1(
            net3, blocks[3:4],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')

    resnet_features = [net, net2, net3, net4]
    print(resnet_features.shape)
    return resnet_features



####mobilenet_v2
def mobilenet_v2_arg_scope(weight_decay, is_training=True, depth_multiplier=1.0, regularize_depthwise=False,
                           dropout_keep_prob=1.0):

    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'center': True, 'scale': True }):

        with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):

            with slim.arg_scope([slim.separable_conv2d],
                                weights_regularizer=depthwise_regularizer, depth_multiplier=depth_multiplier):

                with slim.arg_scope([slim.dropout], is_training=is_training, keep_prob=dropout_keep_prob) as sc:

                    return sc

def mobilenet_v2_224(inp, conv_width=1.4):
    with tf.contrib.slim.arg_scope(mobv.training_scope()):
        net, endpoints = mobv.mobilenet_base(inp, conv_width)
    return net


mobilenet_v2_224.default_image_size = 224


##mobilenetv2_fpn
def conv_hyperparams_fn():
    with contrib_slim.arg_scope([]) as sc:
        return sc

def nearest_neighbor_upsampling(input_tensor, scale=None, height_scale=None,
                                width_scale=None):
  """Nearest neighbor upsampling implementation.

  Nearest neighbor upsampling function that maps input tensor with shape
  [batch_size, height, width, channels] to [batch_size, height * scale
  , width * scale, channels]. This implementation only uses reshape and
  broadcasting to make it TPU compatible.

  Args:
    input_tensor: A float32 tensor of size [batch, height_in, width_in,
      channels].
    scale: An integer multiple to scale resolution of input data in both height
      and width dimensions.
    height_scale: An integer multiple to scale the height of input image. This
      option when provided overrides `scale` option.
    width_scale: An integer multiple to scale the width of input image. This
      option when provided overrides `scale` option.
  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].

  Raises:
    ValueError: If both scale and height_scale or if both scale and width_scale
      are None.
  """
  if not scale and (height_scale is None or width_scale is None):
    raise ValueError('Provide either `scale` or `height_scale` and'
                     ' `width_scale`.')
  with tf.name_scope('nearest_neighbor_upsampling'):
    h_scale = scale if height_scale is None else height_scale
    w_scale = scale if width_scale is None else width_scale
    (batch_size, height, width,
     channels) = shape_utils.combined_static_and_dynamic_shape(input_tensor)
    output_tensor = tf.stack([input_tensor] * w_scale, axis=3)
    output_tensor = tf.stack([output_tensor] * h_scale, axis=2)
    return tf.reshape(output_tensor,
                      [batch_size, height * h_scale, width * w_scale, channels])

def fpn_top_down_feature_maps(image_features,
                              depth,
                              use_depthwise=False,
                              use_explicit_padding=False,
                              use_bounded_activations=False,
                              scope=None,
                              use_native_resize_op=False):
  """Generates `top-down` feature maps for Feature Pyramid Networks.

  See https://arxiv.org/abs/1612.03144 for details.

  Args:
    image_features: list of tuples of (tensor_name, image_feature_tensor).
      Spatial resolutions of succesive tensors must reduce exactly by a factor
      of 2.
    depth: depth of output feature maps.
    use_depthwise: whether to use depthwise separable conv instead of regular
      conv.
    use_explicit_padding: whether to use explicit padding.
    use_bounded_activations: Whether or not to clip activations to range
      [-ACTIVATION_BOUND, ACTIVATION_BOUND]. Bounded activations better lend
      themselves to quantized inference.
    scope: A scope name to wrap this op under.
    use_native_resize_op: If True, uses tf.image.resize_nearest_neighbor op for
      the upsampling process instead of reshape and broadcasting implementation.

  Returns:
    feature_maps: an OrderedDict mapping keys (feature map names) to
      tensors where each tensor has shape [batch, height_i, width_i, depth_i].
  """
  with tf.name_scope(scope, 'top_down'):
    num_levels = len(image_features)
    # image_features [layer4 64,64,24 layer7 32,32,32 layer9 16,16,96 layer19 8,8,1280]
    output_feature_maps_list = []
    output_feature_map_keys = []
    padding = 'VALID' if use_explicit_padding else 'SAME'
    kernel_size = 3
    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d], padding=padding, stride=1):
      top_down = slim.conv2d(
          image_features[-1][1],
          depth, [1, 1], activation_fn=None, normalizer_fn=None,
          scope='projection_%d' % num_levels)
      # top_down 8,8,256
      if use_bounded_activations:
        top_down = tf.clip_by_value(top_down, -ACTIVATION_BOUND,
                                    ACTIVATION_BOUND)
      output_feature_maps_list.append(top_down)
      output_feature_map_keys.append(
          'top_down_%s' % image_features[-1][0])

      for level in reversed(range(num_levels - 1)): # 2,1,0
        if use_native_resize_op:
          with tf.name_scope('nearest_neighbor_upsampling'):
            top_down_shape = shape_utils.combined_static_and_dynamic_shape(
                top_down)
            top_down = tf.image.resize_nearest_neighbor(
                top_down, [top_down_shape[1] * 2, top_down_shape[2] * 2])
        else:
          top_down = nearest_neighbor_upsampling(top_down, scale=2)
        residual = slim.conv2d(
            image_features[level][1], depth, [1, 1],
            activation_fn=None, normalizer_fn=None,
            scope='projection_%d' % (level + 1))
        if use_bounded_activations:
          residual = tf.clip_by_value(residual, -ACTIVATION_BOUND,
                                      ACTIVATION_BOUND)
        if use_explicit_padding:
          # slice top_down to the same shape as residual
          residual_shape = tf.shape(residual)
          top_down = top_down[:, :residual_shape[1], :residual_shape[2], :]
        top_down += residual
        if use_bounded_activations:
          top_down = tf.clip_by_value(top_down, -ACTIVATION_BOUND,
                                      ACTIVATION_BOUND)
        if use_depthwise:
          conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1)
        else:
          conv_op = slim.conv2d
        #if use_explicit_padding:
        #  top_down = ops.fixed_padding(top_down, kernel_size)
        output_feature_maps_list.append(conv_op(
            top_down,
            depth, [kernel_size, kernel_size],
            scope='smoothing_%d' % (level + 1)))
        # output_feature_maps_list: [(8,8,256), (16,16,256), (32,32,256), (64,64,256)]
        output_feature_map_keys.append('top_down_%s' % image_features[level][0])
        # ['top_down_layer_19', 'top_down_layer_14', 'top_down_layer_7', 'top_down_layer_4']
      return collections.OrderedDict(reversed(
          list(zip(output_feature_map_keys, output_feature_maps_list))))


def mobilenet_v2_fpn(inp, is_train, conv_width=1.4):
    ## mobilenet v2
    mindepth = 32 #???
    #pdb.set_trace()   
    #with tf.variable_scope('MobilenetV2') as scope:
    #with tf.contrib.slim.arg_scope(mobv.training_scope()):
    with slim.arg_scope(mobv.training_scope(is_training=is_train, bn_decay=0.9997)):
        net, image_features = mobv.mobilenet_base(inp, final_endpoint='layer_19', depth_multiplier=conv_width)

    # flops
    '''
    #pdb.set_trace()
    g = tf.Graph()
    run_meta = tf.RunMetadata()
    with g.as_default():
        image11 = tf.placeholder(tf.float32, shape = [1, 256, 256, 3])
        feature_maps11 = mobv.mobilenet_base(image11)
        #heatmap_outs11 = self.head_net(feature_maps11[0], is_train)

        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts) 
        print('TF stats gives',flops.total_float_ops/1e9)
    '''
        
    ## fpn
    fpn_max_level = 7 #7
    fpn_min_level=2 #3
    additional_layer_depth=256
    use_depthwise = True
    
    #pdb.set_trace()
    depth_fn = lambda d: max(int(d * conv_width), mindepth)
    with slim.arg_scope(conv_hyperparams_fn()):
        with tf.variable_scope('fpn'):
            feature_blocks = ['layer_4', 'layer_7', 'layer_14', 'layer_19']
            base_fpn_max_level = min(fpn_max_level, 5) #5
            feature_block_list = []
            for level in range(fpn_min_level, base_fpn_max_level + 1):
                feature_block_list.append(feature_blocks[level - 2]) 
        
            # feature_block_list ['layer_4', 'layer_7', 'layer_14', 'layer_19']
            fpn_features = fpn_top_down_feature_maps(
                [(key, image_features[key]) for key in feature_block_list],
                depth=depth_fn(additional_layer_depth),
                use_depthwise=use_depthwise)

            feature_maps = []
            for level in range(fpn_min_level, base_fpn_max_level + 1): 
                feature_maps.append(fpn_features['top_down_{}'.format(
                    feature_blocks[level - 2])]) # 2
            # feature_maps: [(64,64,256), (32,32,256), (16,16,256), (8,8,256)]
            last_feature_map = fpn_features['top_down_{}'.format(
                feature_blocks[base_fpn_max_level - 2])] # 2
            # last_feature_map (8,8,256)

            # Construct coarse features
            padding = 'SAME'
            kernel_size = 3
            for i in range(base_fpn_max_level + 1, fpn_max_level + 1):
                if use_depthwise:
                    conv_op = functools.partial(
                        slim.separable_conv2d, depth_multiplier=1)
                else:
                    conv_op = slim.conv2d

                last_feature_map = conv_op(
                    last_feature_map,
                    num_outputs=depth_fn(additional_layer_depth),
                    kernel_size=[kernel_size, kernel_size],
                    stride=2,
                    padding=padding,
                    scope='bottom_up_Conv2d_{}'.format(i - base_fpn_max_level + 19))
                feature_maps.append(last_feature_map) #add channel 2, 4
    #print(feature_maps) (2,4,8,16,32,64 channel 256)
    return feature_maps

