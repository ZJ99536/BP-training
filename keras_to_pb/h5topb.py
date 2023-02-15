from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

h5_file_path = "/home/zhoujin/learning/model/quad2_m6.h5"
pb_model_path = "/home/zhoujin/learning/model"
"""
冻结模型，可以将训练好的.h5模型文件转成.pb文件
:param h5_file_path: h5模型文件路径
:param pb_model_path: pb模型文件保存路径
:return:
"""
# 加载模型，如有自定义层请参考方法二末尾处如何加载
model = tf.keras.models.load_model(h5_file_path, compile=False)
model.summary()

full_model = tf.function(lambda input_1: model(input_1))
full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=pb_model_path,
                    name="quad2_m6.pb",
                    as_text=False)
print('model has been saved')
