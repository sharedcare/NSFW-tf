#!/usr/bin/env python
"""
Author Yizhen Chen
Deploy on AWS lambda
"""
import numpy as np
import tensorflow as tf
import cv2
import os
import boto3
import base64

s3 = boto3.resource('s3')

FLAGS = None
DATASET_MEAN = [104, 117, 123]

MODEL_GRAPH_DEF_PATH = os.path.join(os.sep, 'tmp', 'model.pb')
MODEL_FILENAME = 'open_nsfw.pb'


def create_graph():
    with tf.gfile.FastGFile(MODEL_GRAPH_DEF_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image, img_size=(256, 256)):
    nparr = np.fromstring(image, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(img_np, img_size)

    H, W, _ = image.shape
    h, w = (224, 224)
    h_off = max((H - h) // 2, 0)
    w_off = max((W - w) // 2, 0)
    image = image[h_off:h_off + h, w_off:w_off + w, :]
    image = image.astype(np.float32, copy=False)
    image -= np.array(DATASET_MEAN, dtype=np.float32)
    feed_input = np.expand_dims(image, axis=0)
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('prob:0')
        data_tensor = sess.graph.get_tensor_by_name('data:0')

        predictions = sess.run(softmax_tensor,
                               feed_dict={data_tensor: feed_input})
        predictions = np.squeeze(predictions)

        print('SFW score: {0:.4f}, NSFW score: {1:.4f}'.format(*predictions))

        return predictions


def lambda_handler(event, context):
    image_b64 = event['body-json']['image']
    if image_b64:
        # This must be called before create_graph().
        print('Downloading Model from S3...')
        s3.Bucket(os.environ['model_bucket_name']).download_file(
            MODEL_FILENAME,
            MODEL_GRAPH_DEF_PATH)
        image_data = base64.b64decode(image_b64)
        predictions = run_inference_on_image(image_data)
        return {
            "statusCode": 200,
            "body": {
                "sfw": predictions[0],
                "nsfw": predictions[1]
            },
            "headers": {
                "Access-Control-Allow-Origin": "*"
            }
        }
    else:
        return {
            "statusCode": 400,
            "body": "image is required",
            "headers": {
                "Access-Control-Allow-Origin": "*"
            }
        }
