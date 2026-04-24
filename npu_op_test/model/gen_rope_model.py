#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from rknn.api import RKNN

tf.compat.v1.disable_eager_execution()

def gen_rope():
    print("Generating RoPE.rknn with flattened 2D input...")
    
    batch, seq_len, num_heads, head_dim = 1, 128, 12, 64
    M = batch * seq_len * num_heads   # 1536
    K = head_dim                      # 64

    x   = tf.compat.v1.placeholder(tf.float32, shape=[M, K], name='x')
    cos = tf.compat.v1.placeholder(tf.float32, shape=[M, K], name='cos')
    sin = tf.compat.v1.placeholder(tf.float32, shape=[M, K], name='sin')

    half = K // 2
    x1 = x[:, :half]
    x2 = x[:, half:]
    c  = cos[:, :half]
    s  = sin[:, :half]
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    output = tf.concat([out1, out2], axis=1)
    output = tf.identity(output, name='rope_output')

    with tf.compat.v1.Session() as sess:
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['rope_output'])
        with open('rope_model.pb', 'wb') as f:
            f.write(frozen.SerializeToString())

    rknn = RKNN()
    rknn.config(target_platform='rk3588')
    ret = rknn.load_tensorflow(tf_pb='rope_model.pb',
                               inputs=['x', 'cos', 'sin'],
                               outputs=['rope_output'],
                               input_size_list=[[M, K], [M, K], [M, K]])
    if ret != 0:
        print("❌ load failed")
        return
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("❌ build failed")
        return
    ret = rknn.export_rknn('RoPE.rknn')
    if ret != 0:
        print("❌ export failed")
        return
    print("✅ RoPE.rknn generated successfully")

if __name__ == '__main__':
    gen_rope()
