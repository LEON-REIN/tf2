# @.@ coding:utf-8 ^-^
# @Author   : Leon Rein
# @Time     : 2020-05-21  ~  20:04 
# @File     : Test.py
# @Software : PyCharm
# @Notice   : It's a WINDOWS version!
#             Just For Test

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
for image, label in mnist_dataset:
    plt.title(label.numpy())
    plt.imshow(image.numpy()[:, :, :])
    plt.show()
    break
