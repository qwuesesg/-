#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from rknn.api import RKNN

tf.compat.v1.disable_eager_execution()

def gen_silu():
    print("Generating silu.rknn...")
    
    # 输入：三维张量 [1, 12, 64]
    input_tensor = tf.compat.v1.placeholder(tf.float32, shape=[1, 12, 64], name='input')
    
    # SiLU = x * sigmoid(x)
    sigmoid_out = tf.sigmoid(input_tensor)
    output = tf.multiply(input_tensor, sigmoid_out, name='silu_output')

    with tf.compat.v1.Session() as sess:
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['silu_output'])
        with open('silu_model.pb', 'wb') as f:
            f.write(frozen.SerializeToString())

    rknn = RKNN()
    rknn.config(target_platform='rk3588')
    ret = rknn.load_tensorflow(tf_pb='silu_model.pb',
                               inputs=['input'],
                               outputs=['silu_output'],
                               input_size_list=[[1, 12, 64]])
    if ret != 0:
        print("❌ load silu_model.pb failed, error:", ret)
        return ret
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("❌ build silu model failed, error:", ret)
        return ret
    ret = rknn.export_rknn('SiLu.rknn')   # 注意：导出为大写 Lu 的文件名
    if ret != 0:
        print("❌ export SiLu.rknn failed, error:", ret)
        return ret
    print("✅ SiLu.rknn generated successfully")
    return 0

if __name__ == '__main__':
    gen_silu()
