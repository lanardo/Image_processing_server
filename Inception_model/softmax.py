import tensorflow as tf
import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing the trainning data")
parser.add_argument("--output_dir", help="path to folder containing the result coef files")
parser.add_argument("--restore", default="yes", help="restore from the checkpoint")

parser.add_argument("--rate", type=float, default=0.0001, help="rate(alpha) for trainning")
parser.add_argument("--epochs", type=int, default=200000, help="max epoches")
parser.add_argument("--strip", type=int, default=50, help="step for writing the result on loop")

a = parser.parse_args()

# a.input_dir = './model'
# a.output_dir = './model'
# a.restore = "no"


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

    learning_rate = a.rate
    in_dir = a.input_dir
    out_dir = a.output_dir
    epochs = a.epochs
    strip = a.strip

    train_data_path = os.path.join(in_dir, 'train_data.csv')
    w_coef_path = os.path.join(out_dir, 'w.csv')
    b_coef_path = os.path.join(out_dir, 'b.csv')
    ckpt_path = os.path.join(out_dir, 'model_bin.ckpt')

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

    """ Loading training data from csv files """
    print('[Step 1] Loading training data ...')
    # for python 2x
    with open(train_data_path) as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        for row in csv_reader:
            x_data.append([float(row[i]) for i in range(0, len(row)-7)])
            y_data.append([float(row[i]) for i in range(len(row)-7, len(row))])

    print("total features    :" + str(len(x_data)))
    print("length of feature :" + str(len(x_data[0])))
    print("length of label   :" + str(len(y_data[0])))

    """ Placeholder """
    print('[Step 2] Placeholder')
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
    print('[Step 3] Initializing the variables.')

    # init = tf.initialize_all_variables()  # python 3x
    init = tf.global_variables_initializer()  # python 2x

    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()

    print(a.restore)
    if a.restore == "yes":
        print('Loading the last learning Session.')
        saver.restore(sess, ckpt_path)

    """ Training cycle """
    print('[Step 4] Training...')
    for step in range(epochs):
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})
        if step % strip == 0:
            ret = sess.run(t1, feed_dict={x: x_data})
            acc1 = acc(sess.run(tf.arg_max(ret, 1)), sess.run(tf.arg_max(y_data, 1))) * 100

            print('    ' + str(step) + ' ' + str(sess.run(cost, feed_dict={x: x_data, y: y_data})) + ' ' + str(acc1))

            saver.save(sess, ckpt_path)

    print('Optimization Finished!')

