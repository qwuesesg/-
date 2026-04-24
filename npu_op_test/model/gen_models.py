import numpy as np
import tensorflow as tf
from rknn.api import RKNN

tf.compat.v1.disable_eager_execution()

def gen_add():
    print("生成 add.rknn...")
    input_a = tf.compat.v1.placeholder(tf.float32, shape=[1,3,224,224], name='input_a')
    input_b = tf.compat.v1.placeholder(tf.float32, shape=[1,3,224,224], name='input_b')
    output = tf.add(input_a, input_b, name='add_output')
    with tf.compat.v1.Session() as sess:
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['add_output'])
        with open('add_model.pb', 'wb') as f:
            f.write(frozen.SerializeToString())
    rknn = RKNN()
    rknn.config(target_platform='rk3588')
    ret = rknn.load_tensorflow(tf_pb='add_model.pb',
                               inputs=['input_a', 'input_b'],
                               outputs=['add_output'],
                               input_size_list=[[1,3,224,224], [1,3,224,224]])
    assert ret == 0, 'load add_model.pb failed'
    ret = rknn.build(do_quantization=False)
    assert ret == 0, 'build add model failed'
    ret = rknn.export_rknn('add.rknn')
    assert ret == 0, 'export add.rknn failed'
    print("add.rknn 生成成功")

def gen_relu():
    print("生成 relu.rknn...")
    input = tf.compat.v1.placeholder(tf.float32, shape=[1,3,224,224], name='input')
    output = tf.nn.relu(input, name='relu_output')
    with tf.compat.v1.Session() as sess:
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['relu_output'])
        with open('relu_model.pb', 'wb') as f:
            f.write(frozen.SerializeToString())
    rknn = RKNN()
    rknn.config(target_platform='rk3588')
    ret = rknn.load_tensorflow(tf_pb='relu_model.pb',
                               inputs=['input'],
                               outputs=['relu_output'],
                               input_size_list=[[1,3,224,224]])
    assert ret == 0, 'load relu_model.pb failed'
    ret = rknn.build(do_quantization=False)
    assert ret == 0, 'build relu model failed'
    ret = rknn.export_rknn('relu.rknn')
    assert ret == 0, 'export relu.rknn failed'
    print("relu.rknn 生成成功")

def gen_conv2d():
    print("生成 conv2d.rknn...")
    # 输入形状 NHWC: [1, 224, 224, 3]
    input = tf.compat.v1.placeholder(tf.float32, shape=[1, 224, 224, 3], name='input')
    kernel = tf.compat.v1.placeholder(tf.float32, shape=[3, 3, 3, 16], name='kernel')
    conv = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='SAME',
                        data_format='NHWC', name='conv')   # 明确 NHWC
    output = tf.identity(conv, name='conv_output')
    with tf.compat.v1.Session() as sess:
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['conv_output'])
        with open('conv2d_model.pb', 'wb') as f:
            f.write(frozen.SerializeToString())
    rknn = RKNN()
    rknn.config(target_platform='rk3588')
    # 输入节点名实际为 input_1（通过 list_nodes.py 确认）
    ret = rknn.load_tensorflow(tf_pb='conv2d_model.pb',
                               inputs=['input_1', 'kernel'],
                               outputs=['conv_output'],
                               input_size_list=[[1,224,224,3], [3,3,3,16]])
    assert ret == 0, 'load conv2d_model.pb failed'
    ret = rknn.build(do_quantization=False)
    assert ret == 0, 'build conv2d model failed'
    ret = rknn.export_rknn('conv2d.rknn')
    assert ret == 0, 'export conv2d.rknn failed'
    print("conv2d.rknn 生成成功")
    
if __name__ == '__main__':
    gen_add()
    gen_relu()
    gen_conv2d()
