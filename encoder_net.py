# !usr/bin/python
# coding=utf-8

"""
构建简单自编码网络
"""

import tensorflow as tf
import argparse
import numpy as np
import matplotlib.pyplot as plt


def weight_init(shape, name):
    return tf.Variable(tf.random_normal(shape), name=name)


def biase_init(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


def train_encoder(x, n_input):
    # 定义网络结构
    n_hidden_1 = 256
    n_hidden_2 = 128

    # 第一个隐含层
    with tf.name_scope('encoder_h1'):
        h1_weight = weight_init([n_input, n_hidden_1], 'weight')
        h1_biase = biase_init([n_hidden_1], 'biase')
        h1 = tf.sigmoid(tf.add(tf.matmul(x, h1_weight), h1_biase))

    # 第二个隐含层
    with tf.name_scope('encoder_h2'):
        h2_weight = weight_init([n_hidden_1, n_hidden_2], 'weight')
        h2_biase = biase_init([n_hidden_2], 'biase')
        h2 = tf.sigmoid(tf.add(tf.matmul(h1, h2_weight), h2_biase))

    # 解码第一个隐含层
    with tf.name_scope('decoder_h1'):
        h1_weight = weight_init([n_hidden_2, n_hidden_1], 'weight')
        h1_biase = biase_init([n_hidden_1], 'biase')
        h1 = tf.sigmoid(tf.add(tf.matmul(h2, h1_weight), h1_biase))

    with tf.name_scope('decoder_h2'):
        h2_weight = weight_init([n_hidden_1, n_input], 'weight')
        h2_biase = biase_init([n_input], 'biase')
        h2 = tf.sigmoid(tf.add(tf.matmul(h1, h2_weight), h2_biase))

    return h2


def main():
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.MakeDirs(FLAGS.model_dir)

    # 搭建网络模型
    n_input = 784
    x = tf.placeholder("float", [None, n_input])
    tf.summary.histogram('x_input', x)

    decoder_out = train_encoder(x, n_input)
    tf.summary.histogram('decoder_out', decoder_out)

    # 定义损失函数优化器
    y_true = x
    cost = tf.reduce_mean(tf.pow(y_true - decoder_out, 2))
    tf.summary.scalar('cost', cost)
    optimizer = tf.train.RMSPropOptimizer(FLAGS.lr).minimize(cost)

    # 定义计数变量,保存迭代次数
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # 保存训练模型
    saver = tf.train.Saver()

    # 设置GPU按需增长
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # 开始模型训练
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(FLAGS.data_dir, fake_data=FLAGS.fake_data)
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init)
        start = global_step.eval()
        merged_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', tf.get_default_graph())
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        total_batches = int(mnist.train.num_examples / FLAGS.batch_size)

        for epoch in range(start, FLAGS.max_steps):
            for i in range(total_batches):
                batch_x, _ = mnist.train.next_batch(FLAGS.batch_size)
                _, s, c = sess.run([optimizer, merged_summaries, cost], feed_dict={x: batch_x})
                train_writer.add_summary(s, i)
            # 打印信息
            if epoch % FLAGS.display_step == 0:
                print("Epoch: ", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            global_step.assign(epoch).eval()
            saver.save(sess, FLAGS.model_dir + "/model.ckpt", global_step=global_step)

        print("Optimization Finished!")

        # # 选择10个测试集进行测试
        s, encode_decode = sess.run([merged_summaries, decoder_out], feed_dict={x: mnist.test.images[:10]})
        # test_writer.add_summary(s, FLAGS.max_steps)
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()


if __name__ == '__main__':
    # 配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False,
                        help='If true, use fake data for unit testing')
    parser.add_argument('--max_steps', type=int, default=40, help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of samples in a mini-batch.')
    parser.add_argument('--display_step', type=int, default=1, help='the steps number for display results.')

    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')

    parser.add_argument('--data_dir', type=str, default="../../MNIST_data/",help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Summaries log directory')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory for storing output models')

    FLAGS, unparsed = parser.parse_known_args()
    main()
