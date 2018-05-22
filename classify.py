#!/usr/bin/env python
"""
Author Yizhen Chen
"""
import numpy as np
import tensorflow as tf
import argparse
import os
import sys

FLAGS = None


def create_graph():
    with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

def run_inference_on_image(filename, img_size=(256, 256)):
    if not tf.gfile.Exists(filename):
        tf.logging.fatal('Image file does not exist %s', filename)

    image_data = tf.gfile.FastGFile(filename, 'rb').read()

def main():
    image = (FLAGS.image_path if FLAGS.image_path else
            'elephant.jpg')
    run_inference_on_image(image)

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to NSFW classification model'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help='Absolute path to image file'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)