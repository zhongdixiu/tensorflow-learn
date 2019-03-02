# ！usr/bin/python
# coding=utf-8

"""
应用TensorFlow实现AlexNet训练MNIST数据集
"""

import tensorflow as tf

# 定义权值
def weight_init(shape, name):
    return tf.Variable(tf.random_normal(shape=shape, name=name))

# 定义卷积操作
def conv2d(name, x, W ,b, strides = 1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)

# 定义池化操作
def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# local response normalization
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)

# 定义AlexNet网络模型
def alex_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    with tf.name_scope('one'):
        conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
        pool1 = maxpool2d('pool1', conv1, k=2)
        norm1 = norm('norm1', pool1, lsize=4)   # 14

    with tf.name_scope('two'):
        conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
        pool2 = maxpool2d('pool2', conv2, k=2)
        norm2 = norm('norm2', pool2, lsize=4)  # 7

    with tf.name_scope('three'):
        conv3 = conv2d('conv2', norm2, weights['wc3'], biases['bc3'])
        pool3 = maxpool2d('pool3', conv3, k=2)
        norm3 = norm('norm3', pool3, lsize=4)

    with tf.name_scope('four'):
        conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])
        conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
        pool5 = maxpool2d('pool5', conv5, k=2)
        norm5 = norm('norm5', pool5, lsize=4)

    with tf.name_scope('fc_one'):
        fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

    with tf.name_scope('fc_two'):
        # fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, dropout)

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

    return out


def run_main():

    # 定义网络超参数
    learning_rate = 0.001
    training_iters = 200000
    batch_size = 128
    display_step = 10

    # 定义网络参数
    n_input = 784
    n_classes = 10
    dropout = 0.75

    # 输入占位符
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    # 定义所有网络参数
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
        'wd1': tf.Variable(tf.random_normal([2*2*256, 4096])),
        'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        'out': tf.Variable(tf.random_normal([4096, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([96])),
        'bc2': tf.Variable(tf.random_normal([256])),
        'bc3': tf.Variable(tf.random_normal([384])),
        'bc4': tf.Variable(tf.random_normal([384])),
        'bc5': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([4096])),
        'bd2': tf.Variable(tf.random_normal([4096])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # 构建模型
    pred = alex_net(x, weights, biases, keep_prob)
    # 损失函数和优化器
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
    tf.summary.scalar('cost', cost)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

    # 衡量矩阵
    correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs/train', tf.get_default_graph())
    # train_writer.close()

    # ================== 开始训练模型 ===========================
    # 设置GPU按需增长
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # 初始化变量
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init)
        step = 1
        # 第一步载入数据
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=True)

        while step*batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, s= sess.run([optimizer, merged_summaries], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            train_writer.add_summary(s, step)

            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1})

                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))

            step += 1
        print("Optimization Finished!")

        # # 计算测试集
        # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
        #                                                          y: mnist.test.labels[:256],
        #                                                      keep_prob: 1}))


if __name__ == '__main__':
    run_main()