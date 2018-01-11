from model.ssd import SSD300
import numpy as np
import tensorflow as tf
import keras as k
from keras import backend as K
from keras.models import Model
import shutil
import os

tf.app.flags.DEFINE_integer('model_version', 1, """Version number of the model.""")
tf.app.flags.DEFINE_string('output_dir', '/tmp/ssd', """Directory where to export inference model.""")
FLAGS = tf.app.flags.FLAGS

sess = tf.Session()
K.set_session(sess)
"""
configuration
"""
input_shape = (300, 300, 3)
num_classes = 21
weights_path = 'weights_SSD300.hdf5'
output_path = os.path.join(tf.compat.as_bytes(FLAGS.output_dir),
                           tf.compat.as_bytes(str(FLAGS.model_version)))

# delete previous folder
if os.path.isdir(FLAGS.output_dir):
    shutil.rmtree(FLAGS.output_dir)

model = SSD300(input_shape, num_classes=num_classes)

model.load_weights(weights_path)

from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"images": model.input}, {"prediction": model.output})

save_model_builder = builder.SavedModelBuilder(output_path)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

save_model_builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], 
                                     signature_def_map={
                                         'predict_images': prediction_signature,
                                     },
                                     legacy_init_op=legacy_init_op)
save_model_builder.save()
