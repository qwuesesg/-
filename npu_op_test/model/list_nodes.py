import tensorflow as tf

graph_def = tf.compat.v1.GraphDef()
with open('conv2d_model.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())

for node in graph_def.node:
    print(node.name)
