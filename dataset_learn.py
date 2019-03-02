# coding=utf-8

"""以MNIST为例, 使用dataset API

"""

import tensorflow as tf
import numpy as np

def data_parser(record):
    """将tfrecord格式数据解析,获得data和label

    :param record: TFRecordDataset
    :return: image and label
    """
    # 生成record时,相应的键值对
    # FixedLenFeature(shape, dtype, default_value)
    # default_vlue: Value to be used if an example is missing this feature. It
    #         must be compatible with `dtype` and of the specified `shape`.
    keys_to_features = {
        "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
        "image/format": tf.FixedLenFeature((), tf.string, default_value="raw"),
        'image/height': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/width': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }

    parsed = tf.parse_single_example(record, keys_to_features)
    # 解析得到的数据是string类型,进行转换成为numpy
    imgs = tf.image.decode_png(parsed['image/encoded'], dtype=tf.uint8)
    imgs = tf.reshape(imgs, [28, 28, 1])
    labels = tf.cast(parsed['image/class/label'], tf.int32)
    labels = tf.one_hot(labels, NUM_CLASS)

    print("IMAGES", imgs)
    print("LABELS", labels)

    return {"img_raw": imgs}, labels


# 定义数据库参数
EPOCHS = 2
NUM_CLASS = 10


file_name = './mnist_train.tfrecord'
# 构建数据库
dataset = tf.data.TFRecordDataset(file_name)
print('DATASET', dataset)

# 数据转换,获得batch迭代数据
dataset = dataset.map(data_parser)
print("DATASET1", dataset)
dataset = dataset.shuffle(buffer_size=100)
print("DATASET2", dataset)
dataset = dataset.batch(8)
print("DATASET3", dataset)
dataset = dataset.repeat(EPOCHS)
print("DATASET4", dataset)
iterator = dataset.make_one_shot_iterator()

feature, label = iterator.get_next()
print("FEATURE", feature)
print("LABEL", label)

with tf.Session() as sess:
    print("SESS RUN", sess.run(label))
