import numpy as np
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Normalization
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

pb_file_path = "/home/zhoujin/learning/model/quad2_m6.pb"
dataset = loadtxt('/home/zhoujin/trajectory-generation/trajectory/quad2.txt', delimiter=',')
# split into input (X) and output (y) variables

X = dataset[:,0:15]
y = dataset[:,18:36]

# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit(X)
# X = min_max_scaler.transform(X)

# min_max_scaler.fit(y)
# remember = min_max_scaler
# y = min_max_scaler.transform(y)

input = X[61733:61794,0:15]
output = dataset[61733:61794,18:36]
output_origin = y[61733:61794,:]

with tf.compat.v1.Session() as sess:
    with tf.io.gfile.GFile(pb_file_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.compat.v1.import_graph_def(graph_def)
    # print all operation names
    # for op in sess.graph.get_operations():
    #     print(op.name)
    # 输入
    input_x = sess.graph.get_tensor_by_name('import/input_1:0')
    # 输出
    # output = sess.graph.get_tensor_by_name('import/output:0')
    # 预测结果
    ynew = sess.run({input_x: input})




plt.plot(ynew[:,0:1])
# plt.plot(output_origin[:,6:7])
plt.plot(output[:,0:1])
plt.show()
# print(remember.inverse_transform(ynew))
# print(output)

