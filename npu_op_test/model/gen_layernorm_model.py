#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from rknn.api import RKNN

tf.compat.v1.disable_eager_execution()

def gen_layernorm():
    print("Generating layernorm.rknn...")
    # 输入: [1, 12, 64]
    input_tensor = tf.compat.v1.placeholder(tf.float32, shape=[1, 12, 64], name='input')
    
    # 手动计算 LayerNorm (无 affine 参数)
    mean = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    variance = tf.reduce_mean(tf.square(input_tensor - mean), axis=-1, keepdims=True)
    epsilon = 1e-5
    normed = (input_tensor - mean) / tf.sqrt(variance + epsilon)
    output = tf.identity(normed, name='layernorm_output')

    with tf.compat.v1.Session() as sess:
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['layernorm_output'])
        with open('layernorm_model.pb', 'wb') as f:
            f.write(frozen.SerializeToString())

    rknn = RKNN()
    rknn.config(target_platform='rk3588')
    ret = rknn.load_tensorflow(tf_pb='layernorm_model.pb',
                               inputs=['input'],
                               outputs=['layernorm_output'],
                               input_size_list=[[1, 12, 64]])
    if ret != 0:
        print("load layernorm_model.pb failed")
        return ret
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("build layernorm model failed")
        return ret
    ret = rknn.export_rknn('layernorm.rknn')
    if ret != 0:
        print("export layernorm.rknn failed")
        return ret
    print("layernorm.rknn generated successfully")
    return 0

if __name__ == '__main__':
    gen_layernorm()
