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


'''
It shows what Dataset contains, as well as some methods -- from_tensor_slices(), take()
                                                           batch(), shuffle(), prefetch()  
'''
# dataset = tf.data.Dataset.from_tensor_slices(
#     {
#         "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
#         "b": np.random.uniform(size=(5, 3))
#     })

# dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2)))

dataset = tf.data.Dataset.from_tensor_slices(
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.random.uniform(size=(5, 2))))
dataset = dataset.batch(3)
# dataset = dataset.shuffle(buffer_size=2)
# dataset = dataset.prefetch(2)
dataset = dataset.take(1)  # Only take 1 batch
for a, b in dataset.enumerate():
    print('?????', a, "HHH", b)
print('*****************')
aa = list(dataset.as_numpy_iterator())
print(aa)

