import tensorflow as tf
from rknn.api import RKNN

tf.compat.v1.disable_eager_execution()

def gen_softmax():
    print("生成 softmax.rknn...")
    input = tf.compat.v1.placeholder(tf.float32, shape=[1, 12, 64], name='input')
    output = tf.nn.softmax(input, axis=-1, name='softmax_output')
    with tf.compat.v1.Session() as sess:
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['softmax_output'])
        with open('softmax_model.pb', 'wb') as f:
            f.write(frozen.SerializeToString())
    rknn = RKNN()
    rknn.config(target_platform='rk3588')
    ret = rknn.load_tensorflow(tf_pb='softmax_model.pb',
                               inputs=['input'],
                               outputs=['softmax_output'],
                               input_size_list=[[1, 12, 64]])
    assert ret == 0, 'load softmax_model.pb failed'
    ret = rknn.build(do_quantization=False)
    assert ret == 0, 'build softmax model failed'
    ret = rknn.export_rknn('softmax.rknn')
    assert ret == 0, 'export softmax.rknn failed'
    print("softmax.rknn 生成成功")

if __name__ == '__main__':
    gen_softmax()
