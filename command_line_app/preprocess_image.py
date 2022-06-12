import tensorflow as tf
import numpy as np

def process_image(image, image_size = 224):
    image = np.asarray(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    image = image.numpy()
    return image
