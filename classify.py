#!/usr/bin/env python
"""
Author Yizhen Chen
"""
import numpy as np
import tensorflow as tf
import argparse
import os
import sys
import cv2
import skimage
import skimage.io
from PIL import Image
from io import BytesIO

FLAGS = None
DATASET_MEAN = [104, 117, 123]


def create_graph():
    with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(filename, img_size=(256, 256)):
    if not tf.gfile.Exists(filename):
        tf.logging.fatal('Image file does not exist %s', filename)

    image = cv2.imread(filename)
    image = cv2.resize(image, img_size)

    H, W, _ = image.shape
    h, w = (224, 224)
    h_off = max((H - h) // 2, 0)
    w_off = max((W - w) // 2, 0)
    image = image[h_off:h_off + h, w_off:w_off + w, :]
    image = image.astype(np.float32, copy=False)
    image -= np.array(DATASET_MEAN, dtype=np.float32)
    input = np.expand_dims(image, axis=0)
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('prob:0')
        data_tensor = sess.graph.get_tensor_by_name('data:0')

        predictions = sess.run(softmax_tensor,
                               feed_dict={data_tensor: input})
        predictions = np.squeeze(predictions)

        print('SFW score: {0:.3f}, NSFW score: {1:.3f}'.format(*predictions))

    return predictions

def main(_):
    image = (FLAGS.image_path if FLAGS.image_path else
            'elephant.jpg')
    run_inference_on_image(image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default='./nsfw_model/open_nsfw.pb',
        help='Path to NSFW classification model'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help='Absolute path to image file'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)