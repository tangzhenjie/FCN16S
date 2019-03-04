__author__ = 'tangzhenjie'
import os
from tensorflow.python.platform import gfile
from six.moves import cPickle as pickle
import glob
import random

import TensorflowUtils as utils

# 数据集下载URL
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'

"""
从.pickle文件读取训练集和验证机文件名数组

param：
    data_dir: 文件存放的文件夹
return：
    训练集和验证机文件名数组 （tuple）
"""
def read_dataset(data_dir):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)

    # 验证文件如果不存在就去下载
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        #下载并解压好文件后获取训练集合验证集文件名数组

        SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
        print("序列化 ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records
def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("\\")[-1])[0]
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list

