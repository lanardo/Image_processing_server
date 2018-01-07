import tensorflow as tf
import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", help="path to folder containing the model checkpoint")
a = parser.parse_args()


def xaver_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def acc(d1, d2):
    cnt = 0
    for i in range(d1.__len__()):
        if d1[i] == d2[i]:
            cnt += 1

    return float(cnt)/d1.__len__()


def sel_max(data):
    ret_ind = []
    for i in range(data.__len__()):
        if data[i][0] == 1:
            ret_ind.append(0)
        else:
            ret_ind.append(1)

    return ret_ind


if __name__ == '__main__':

    learning_rate = 0.0004
    ckpt_dir = a.ckpt_dir

    w_coef_path = os.path.join(ckpt_dir, 'w.csv')
    b_coef_path = os.path.join(ckpt_dir, 'b.csv')
    ckpt_path = os.path.join(ckpt_dir, 'model_bin.ckpt')

    labels = ['front', 'front_3_quarter', 'side', 'rear_3_quarter', 'rear', 'interior', 'tire']
    directions = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ]

    x_data = []
    y_data = []

    """ Placeholder """
    print('Placeholder')
    x = tf.placeholder('float', [None, 2048])  # len(feature) = 2048
    y = tf.placeholder('float', [None, 7])  # len(Directions) = 7 : classes

    W1 = tf.get_variable('W1', shape=[2048, 7], initializer=xaver_init(2048, 7))
    b1 = tf.Variable(tf.zeros([7]))
    activation = tf.add(tf.matmul(x, W1), b1)
    t1 = tf.nn.softmax(activation)

    """ Minimize error using cross entropy """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=activation, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Gradient Descent

    """ Initializing the variables """
    print('Initializing the variables.')

    # init = tf.initialize_all_variables()  # python 3x
    init = tf.global_variables_initializer()  # python 2x

    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()

    print('Loading the last learning Session.')
    saver.restore(sess, ckpt_path)

    ret_W = sess.run(W1)
    ret_b = sess.run(b1)

    print('Saving the coefs W, b')
    # print(len(ret_W), len(ret_W[0]))
    # print(len(ret_b))

    # with open(w_coef_path, 'w', newline='') as fp:  # for python 3x
    with open(w_coef_path, 'wb') as fp:  # for python 2x
        wr = csv.writer(fp, delimiter=',')
        wr.writerows(ret_W)
    # with open(    b_coef_path, 'w', newline='') as fp:  # for python 3x
    with open(b_coef_path, 'wb') as fp:  # for python 2x
        wr = csv.writer(fp, delimiter=',')
        wr.writerow(ret_b)

    print('Saving coeficient files Finished!')

