# coding = utf-8
"""将minist数据记录为record格式

"""

import os
import gzip
import sys
import numpy as np
import tensorflow as tf

def read_imgs(filename, num_images):
    """读入图片数据

    :param filename:
    :param num_images:
    :return:
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            28 * 28 * num_images * 1)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, 28, 28, 1)
    return data


def read_labels(filename, num_labels):
    """读入图片标签数据

    :param filename:
    :param num_labels:
    :return:
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return labels


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def add_to_record(data_filename, label_filename, num_images, tf_writer):
    """将数据写入tfrecord文件

    :param data_filename: 数据文件路径
    :param label_filename: 标签文件路径
    :param num_images: 数据数目
    :param tf_writer: tfrecord文件写指针
    :return:
    """
    imgs = read_imgs(data_filename, num_images)
    labels = read_labels(label_filename, num_images)

    shape = (28, 28, 1)
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)
        # example_img = tf.image.decode_png(encoded_png)

        with tf.Session() as sess:
            for j in range(num_images):
                sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
                sys.stdout.flush()

                # png_string, exa_img = sess.run([encoded_png, example_img], feed_dict={image: imgs[j]})
                png_string = sess.run(encoded_png, feed_dict={image: imgs[j]})

                example = image_to_tfexample(png_string, 'png'.encode(), 28, 28, labels[j])
                tf_writer.write(example.SerializeToString())


DATA_DIR = '/media/xiu/data/MNIST_data/'
TRAIN_DATA_FILENAME = 'train-images-idx3-ubyte.gz'
TRAIN_LABEL_FILENAME = 'train-labels-idx1-ubyte.gz'
TEST_DATA_FILENAME = 't10k-images-idx3-ubyte.gz'
TEST_LABEL_FILENAME = 't10k-labels-idx1-ubyte.gz'

training_filename = './mnist_%s.tfrecord' % 'train'
testing_filename = './mnist_%s.tfrecord' % 'test'

# 转化训练数据
with tf.python_io.TFRecordWriter(training_filename) as tf_writer:
    data_filename = os.path.join(DATA_DIR, TRAIN_DATA_FILENAME)
    label_filename = os.path.join(DATA_DIR, TRAIN_LABEL_FILENAME)
    add_to_record(data_filename, label_filename, 60000, tf_writer)

# 转化测试数据
with tf.python_io.TFRecordWriter(testing_filename) as tf_writer:
    data_filename = os.path.join(DATA_DIR, TEST_DATA_FILENAME)
    label_filename = os.path.join(DATA_DIR, TEST_LABEL_FILENAME)
    add_to_record(data_filename, label_filename, 10000, tf_writer)
