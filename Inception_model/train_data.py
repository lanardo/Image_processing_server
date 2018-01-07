import tensorflow as tf
import os
import numpy as np
import sys
import tarfile
import cv2
import csv
import argparse

from six.moves import urllib


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--output_dir", help="path to folder containing combined/output image")
parser.add_argument("--mode", help="crate the train_data or inference a image", default='train')
a = parser.parse_args()

a.input_dir = '../data/samples'
# a.output_dir = 'Inception_model/model'
a.output_dir = './model'
a.mode = 'train'

MODEL_DIR = a.output_dir

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
LABELS = ['front', 'front_3_quarter', 'side', 'rear_3_quarter', 'rear', 'interior', 'tire']
directions = [
    [1., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 1.]
]
train_data_path = os.path.join(MODEL_DIR, 'train_data.csv')


def create_graph():

    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = MODEL_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def inference(in_dir, out_dir):
    pass


def run_inference_on_image(img_path):

    """
    if not tf.gfile.Exists(img_path):
        tf.logging.fatal('File does not exist %s', img_path)
    image_data = tf.gfile.FastGFile(img_path, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        # 'pool_3:0':  2048 float
        softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        return predictions
    """

sess, softmax_tensor = None, None


def get_feature_from_image(img_path):
    """Runs extract the feature from the image.
        Args: img_path: Image file name.

    Returns:  predictions: 2048 * 1 feature vector
    """
    global sess, softmax_tensor

    if not tf.gfile.Exists(img_path):
        tf.logging.fatal('File does not exist %s', img_path)
    image_data = tf.gfile.FastGFile(img_path, 'rb').read()

    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)
    return predictions


def create_feature_csv(dir_path, label):

    sys.stdout.write("\n [Scanning folder]: " + dir_path + '\n')
    features = []

    count = 0
    # valid_ext = ['.jpg', '.png']
    for f in os.listdir(dir_path):

        fn, ext = os.path.splitext(f)

        if ext.lower() == '.png':
            # # Convert image file to jpg
            # im = cv2.imread(os.path.join(dir_path, f))
            #
            # out_fn = os.path.join(dir_path, fn + '.jpg')
            # cv2.imwrite(out_fn, im)
            # os.remove(os.path.join(dir_path, f))
            continue
        elif ext.lower() == '.jpg':
            out_fn = os.path.join(dir_path, f)

        # Extract the feature vector per each image
        feature = get_feature_from_image(out_fn)
        line = feature.tolist()
        line.extend(directions[LABELS.index(label)])
        features.append(line)
        count += 1

        # for only testing of local with 3 images per each class
        # if count == 3:
        #     break

        sys.stdout.write('\r' + str(count) + "|" + f)
        sys.stdout.flush()

    return features


def train_data(in_dir):
    global sess, softmax_tensor

    maybe_download_and_extract()

    # Creates graph from saved GraphDef.
    create_graph()
    sess = tf.Session()
    softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')

    total_features = []

    for label in LABELS:
        label_path = os.path.join(in_dir, label + '/')
        if not os.path.isdir(label_path):
            raise Exception("There is no such directory", label_path)

        features = create_feature_csv(label_path, label)

        total_features.extend(features)

    # with open(train_data_path, 'w', newline='') as fp:  # for python 3x
    with open(train_data_path, 'wb') as fp:  # for python 2x
        wr = csv.writer(fp, delimiter=',')
        wr.writerows(total_features)

if __name__ == '__main__':

    if a.input_dir is None:
        raise Exception("Input dir not defined")
    input_dir = a.input_dir
    sys.stdout.write("Input dir : " + input_dir + '\n')

    if a.output_dir is None:
        raise Exception("Output dir not defined")
    output_dir = a.output_dir
    sys.stdout.write("Output dir : " + output_dir + '\n')

    if a.mode is None:
        raise Exception("Mode not defined")
    mode = a.mode
    if mode == "train":
        sys.stdout.write("+---------------------------------------------------------+ \n"
                         "|   Create the CSV files for training Classifier Model.   | \n"
                         "+---------------------------------------------------------+ \n")
        train_data(input_dir)

    elif mode == "inference":
        sys.stdout.write("+---------------------------------------------------------+ \n"
                         "|   Inference the image with built model.                 | \n"
                         "+---------------------------------------------------------+ \n")
        inference(input_dir, output_dir)

    sys.stdout.write("\n Done in mode(" + mode + ")successfully!.")
