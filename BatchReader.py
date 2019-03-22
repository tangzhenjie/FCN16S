import tensorflow as tf
import read_MITSceneParsingData as Reader
import numpy as np

#dataset_dir = "D:\pycharm_program\FCN16S\Data_zoo\MIT_SceneParsing\\"


#train_filepaths, eval_filepaths = Reader.read_dataset(dataset_dir)
#train_filepaths = tf.convert_to_tensor(train_filepaths, dtype=tf.string)
#i = 0
#train_filepaths = np.array(train_filepaths)
#train_filepaths1 = train_filepaths[:, 1]
#print(train_filepaths1[0])


"""
    读取batch数据 
 :param  image_filepaths tensor dtype=string 图像路径
          annotation_filepaths   tensor dtype=string 标签图像路径
          image_size   图像剪裁大小
          batch_size   batch大小
 :return  tuple  
"""
def read_batch_image(image_filepaths, label_filepaths, image_size, batch_size=2):
    image, label = tf.train.slice_input_producer([image_filepaths, label_filepaths], shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize images to a common size
    image = tf.image.resize_images(image, [image_size, image_size])

    # Normalize(后期改动)
    #image = image * 1.0 / 127.5 - 1.0

    # Read labels from disk
    label = tf.read_file(label)
    label = tf.image.decode_png(label, channels=1)

    # Resize labels to a common size
    label = tf.image.resize_images(label, [image_size, image_size])

    X, Y = tf.train.batch([image, label], batch_size=batch_size, capacity=batch_size * 8, num_threads=4)
    return X, Y