from __future__ import print_function

import sys

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import json

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from scipy.misc import imread
from scipy.misc import imresize
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from utils.ssd_utils import BBoxUtility

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']

bbox_util = BBoxUtility(21)

def _decode_results(results):
    results = np.array(results)
    det_label = results[0, :, 0]
    det_conf = results[0, :, 1]
    det_xmin = results[0, :, 2]
    det_ymin = results[0, :, 3]
    det_xmax = results[0, :, 4]
    det_ymax = results[0, :, 5]

    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    
    data = []
    for i in range(top_conf.shape[0]):
        _data = {}
        label = int(top_label_indices[i])
        score = top_conf[i]
        _data['label'] = voc_classes[label-1]
        _data['xmin'] = top_xmin[i]
        _data['ymin'] = top_ymin[i]
        _data['xmax'] = top_xmax[i]
        _data['ymax'] = top_ymax[i]
        _data['score'] = score
        _data['label_id'] = label-1
        data.append(_data)
    return data

def do_inference(hostport, workdir):
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # image
    sys.stdout.write('read image...\n')
    sys.stdout.flush()
    
    img = image.load_img('my-test-motor-man.jpg', target_size=(300, 300))
    img = image.img_to_array(img)
    inputs = []
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))
    
    # predict
    sys.stdout.write('predicting...\n')
    sys.stdout.flush()
    
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'ssd'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(inputs, shape=inputs.shape))
    results = stub.Predict(request, 10.0)
    results = results.outputs['prediction'].float_val
    results = np.array(results)
    results = np.reshape(results, (1, -1, 33))
    results = bbox_util.detection_out(results)
    
    return _decode_results(results)

def main(_):
    server = 'localhost:9000'
    workdir = '/tmp'
    objects = do_inference(server, workdir)
    print(json.dumps(objects, indent=1))

if __name__ == '__main__':
    tf.app.run()
