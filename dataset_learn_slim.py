# coding=utf-8

"""以MNIST为例,使用slim.data
"""

import os
import tensorflow as tf

slim = tf.contrib.slim

def get_data(data_dir, num_samples, num_class, file_pattern='*.tfrecord'):
    """返回slim.data.Dataset

    :param data_dir: tfrecord文件路径
    :param num_samples: 样本数目
    :param num_class: 类别数目
    :param file_pattern: tfrecord文件格式
    :return:
    """
    file_pattern = os.path.join(data_dir, file_pattern)
    keys_to_features = {
        "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
        "image/format": tf.FixedLenFeature((), tf.string, default_value="raw"),
        'image/height': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/width': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }
    items_to_handlers = {
        "image": slim.tfexample_decoder.Image(channels=1),
        "label": slim.tfexample_decoder.Tensor("image/class/label")
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    items_to_descriptions = {
        "image": 'A color image of varying size',
        "label": 'A single interger between 0 and ' + str(num_class - 1)
    }

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=items_to_descriptions,
        num_classes=num_class,
        label_to_names=label_to_name
    )


NUM_EPOCH = 2
BATCH_SIZE = 8
NUM_CLASS = 10
NUM_SAMPLE = 60000

label_to_name = {'0': 'one', '1': 'two', '3': 'three', '4': 'four', '5': 'five',
                 '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
data_dir = './'
dataset = get_data(data_dir, NUM_SAMPLE, NUM_CLASS, 'mnist_train.tfrecord')
data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = data_provider.get(['image', 'label'])

# 组合数据
images, labels = tf.train.batch([image, label], batch_size=BATCH_SIZE)
labels = slim.one_hot_encoding(labels, NUM_CLASS)

