__author__ = 'tangzhenjie'
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io
import tensorflow as tf
import scipy.misc as misc

"""
下载对应url的文件

param： 
    dir_path: 下载和解压文件的位置
    url_name: 要下载的文件的url
    is_tarfile: 是不是tar文件
    is_zipfile: 是不是zip文件
"""
def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    #首先验证要下载到的解压到的文件夹是否是存在
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 判断有没有下载，没有再去下载
    file_name = url_name.split('/')[-1]
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(file_path):
        # 定义一个下载过程中显示进度的函数
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (file_name, float(count * block_size) / float(total_size) * 100.0)
            )
            # 刷新输出
            sys.stdout.flush()
        file_path, _ = urllib.request.urlretrieve(url_name, file_path, reporthook=_progress)

        # 获取文件信息
        statinfo = os.stat(file_path)
        print('Succesfully downloaded', file_name, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(file_path, 'r:gz').extractall(dir_path)
        if is_zipfile:
            with zipfile.ZipFile(file_path) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)
"""
获取模型数据

:param dir_path    下载的位置
       model_url    模型的网络位置
"""
def get_model_data(dir_path, model_url):
    maybe_download_and_extract(dir_path, model_url)

    # 判断是否下载下来
    filename = model_url.split("/")[-1]
    file_path = os.path.join(dir_path, filename)
    if not os.path.exists(file_path):
        raise IOError("VGG16 model not found")
    data = scipy.io.loadmat(file_path)

    return data

# 有权重初始值定义在网络中生成变量的函数
def get_variable(weights, name):
    # 定义常数初始化器
    init = tf.constant_initializer(weights, dtype=tf.float32)
    # 生成变量
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var

# 有变量的shape生成平均值为0标准差为0.02的截断的正态分布数值的变量
def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

# 生成b值的变量
def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


####################下面定义操作#########################

# 定义卷积输入和输出大小不变（通道可能变化）操作
def conv2d_basic(x, W, bias):
    # stride 1 padding same保证卷积输入和输出相同
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    return tf.nn.bias_add(conv, bias)

# 定义卷积输出是输入的二分之一
def conv2d_strided(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

# 定义maxpool层使图像缩小一半
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2 , 1], strides=[1, 2, 2, 1], padding="SAME")

# 定义平均池化使图像缩小一半
def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

######################图像处理方法#######################
def process_image(image, mean_pixel):
    return image - mean_pixel

def unprocess_image(image, mean_pixel):
    return image + mean_pixel

#####################tensorbord处理方法#################
def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))

def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))

def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)

#######################padding操作####################
# 因为官方caffe代码说是先padding100
def pading(image, paddingdata):
    if len(image.shape) == 3:
        # tensor的shape为[height, width, channels]
        target_height = image.shape[0] + paddingdata * 2
        target_width = image.shape[1] + paddingdata * 2
        return tf.image.pad_to_bounding_box(image,offset_height=paddingdata, offset_width=paddingdata, target_height=target_height,target_width=target_width)
    elif len(image.shape) == 4:
        # [batch, height, width, channels]
        target_height = image.shape[1] + paddingdata * 2
        target_width = image.shape[2] + paddingdata * 2
        return tf.image.pad_to_bounding_box(image, offset_height=paddingdata, offset_width=paddingdata, target_height=target_height,target_width=target_width)
    else:
        raise ValueError("image tensor shape error")

# 反卷积操作
def conv2d_transpose_strided(x, w, b, output_shape=None, stride=2):
    if output_shape is None:
        # 如果默认就让反卷积的输出图片大小扩大一倍，通道为卷积核上的输出通道
        tmp_shape = x.get_shape().as_list()
        tmp_shape[1] *= 2
        tmp_shape[2] *= 2
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], tmp_shape[1], tmp_shape[2], w.get_shape().as_list()[2]])
    conv = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding="SAME")

    return tf.nn.bias_add(conv, b)

# 保存图像
def save_image(image, save_dir, name, mean=None):
    """
    Save image by unprocessing if mean given else just save
    :param image:
    :param save_dir:
    :param name:
    :param mean:
    :return:
    """
    if mean:
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name + ".png"), image)