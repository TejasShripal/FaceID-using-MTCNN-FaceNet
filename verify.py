import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print("Version of Tensorflow: ", tf.__version__)
print("Cuda Availability: ", tf.test.is_built_with_cuda())
print("GPU  Availability: ", tf.config.list_physical_devices('GPU'))
print(tf. __version__)
