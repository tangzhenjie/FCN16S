"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""

import numpy as np
import scipy.misc as misc

class BatchDatset:
    files = [] # 存放图像文件路径
    images = [] # 存放图像数据数组
    annotations = [] # 存放标签图s像数据
    image_options = {} # 改变图像的选择
    batch_offset = 0 # 获取batch数据开始的偏移量
    epochs_completed = 0 # 记录epoch的次数
    """
    Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
    """
    # 构造函数
    def __init__(self, record_list, image_options={}):
        print("Initializing Batch Dataset Reader...")
        print(image_options)

        self.files = record_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self._channels = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self._channels = False
        self.annotations = np.array([np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])

        print(self.images.shape)
        print(self.annotations.shape)

    def _transform(self, filename):
        # 读取图像数据到ndarray
        image = misc.imread(filename)
        # 保证图像通道数为3
        if self._channels and len(image.shape) < 3:
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image, [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image
        return np.array(resize_image)

    # 获取全部的图像和标记图像
    def get_records(self):
        return self.images, self.annotations

    # 修改偏移量
    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    # 获取下一个batch
    def next_batch(self, batch_size):
        # 开始位置
        start = self.batch_offset

        # 下一个batch的开始位置（也是这次的结束位置）
        self.batch_offset += batch_size

        # 判断位置是否超出界限
        if self.batch_offset > self.images.shape[0]:
            # 超出界限证明完成一次epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")

            # 准备下一次数据
            # 首先打乱数据
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]

            # 开始下一次epoch
            start = 0
            self.batch_offset = batch_size

        # 生成数据
        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    # 获取一组随机的batch
    def get_random_batch(self, batch_size):
        indexs = np.random.randint(0, self.images.shape[0], size=batch_size).tolist()
        return self.images[indexs], self.annotations[indexs]

