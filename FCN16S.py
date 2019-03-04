from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange # 兼容python2和python3


# 定义一些网络需要的参数(可以以命令行可选参数进行重新赋值)
FLAGS = tf.flags.FLAGS
# batch大小
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
# 定义日志文件位置
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
# 定义图像数据集存放的路径
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to the dataset")
# 定义学习率
tf.flags.DEFINE_float("learning_rate", "1e-4", "learning rate for Adam Optimizer")
# 存放VGG16模型的mat (我们使用matlab训练好的VGG16参数)
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
# 是否是调试状态（如果是调试状态会额外保存一些信息）
tf.flags.DEFINE_bool("debug", "False", "Model Debug:True/ False")
# 执行的状态（训练 测试 显示）
tf.flags.DEFINE_string("mode", "train", "Mode: train/ test/ visualize")

# 模型地址
MODEL_URL = "http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat"

# 最大迭代次数
MAX_ITERATION = int(1e5 + 1)
# MIT数据集的类别数
NUM_OF_CLASSES = 151
# 首先VGG16网络中的图像输入224*224(但是我们这个网络理论上可以输入任意图片大小）
IMAGE_SIZE = 224

"""
 首先定义该网络与VGG16相同的部分
 :param weight 从.mat中获得的权重
        image  网络输入的图像
"""
def vgg_net(weights, image):
    # 首先我们定义FCN16S中使用VGG16层中的名字，用来生成相同的网络
    layers = (
        "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
        "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
        "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "pool3",
        "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3" "pool4",
        "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "pool5"
    )
    # 生成的公有层的所有接口
    net = {}
    # 当前输入
    current = image

    for i, name in enumerate(layers):
        # 获取前面层名字的前四个字符
        kind = name[:4]
        if kind == "conv":
            kernels = weights[i][0][0][0][0][0]
            bias = weights[i][0][0][0][0][1]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            # 生成变量
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == "relu":
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == "pool":
            current = utils.avg_pool_2x2(current)\

        net[name] = current
    return net










model_data = utils.get_model_data("D:\pycharm_program\FCN16S\VGG16MODEL", MODEL_URL)
layers = model_data["layers"]
vgg_layers = model_data["layers"][0] # type 1*37 (37层）

for element in xrange(0, 37):
    layer = vgg_layers[element]
    struct = layer[0][0]
    number = len(struct)
    if number == 5:
        # weights pad type name stride
        print(struct[3])
    if number == 2:
        # relu层信息
        print(struct[1])
    if number == 6:
        # pool层信息或者是最后一层信息
        print(struct[0])


