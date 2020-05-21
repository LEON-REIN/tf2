# @.@ coding: utf-8 ^-^
# @Author   : Leon Rein
# @Time     : 2020-05-20  ~  20:14 
# @File     : cifar2.py
# @Software : PyCharm
# @Notice   : It's a WINDOWS version!
#             训练一个模型来对飞机airplane和机动车automobile两种图片进行分类
#             ipython 编码问题: 执行 `!chcp 65001` 可以添加至 Setting-Console-Python Console-Starting Script


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot as plt

'''
1. Load datasets-sets && 2. Data Processing
'''

BATCH_SIZE = 100


def load_image(img_path, size=(32, 32)):
    labels = tf.constant(1, tf.int8) if tf.strings.regex_full_match(img_path, ".*automobile.*") \
        else tf.constant(0, tf.int8)  # automobile labeled as 1, while airplane represents 0
    imgs = tf.io.read_file(img_path)
    imgs = tf.image.decode_jpeg(imgs)  # jpeg format!
    imgs = tf.image.resize(imgs, size) / 255.0  # Resize images to 'size' and normalization
    return imgs, labels


# 使用并行化预处理 num_parallel_calls 和预存数据 prefetch 来提升性能
ds_train = tf.data.Dataset.list_files(r"datasets\cifar2_datasets\train\*\*.jpg") \
    .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

ds_test = tf.data.Dataset.list_files(r"datasets\cifar2_datasets\test\*\*.jpg") \
    .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

# Show Some Pictures
# plt.figure(figsize=(8, 8))
# for i, (img, label) in enumerate(ds_train.unbatch().take(9)):
#     ax = plt.subplot(3, 3, i + 1)
#     ax.imshow(img.numpy())
#     ax.set_title("label = %d" % label)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()

# for x, y in ds_train.take(1):
#     print(x.shape, y.shape)  # (100, 32, 32, 3) (100,)

'''
3. Keras Modeling (Using  Functional APIs)
'''

tf.keras.backend.clear_session()  # Clear sessions

inputs = layers.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, kernel_size=(3, 3))(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, kernel_size=(5, 5))(x)
x = layers.MaxPool2D()(x)
x = layers.Dropout(rate=0.1)(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=inputs, outputs=outputs)

# model.summary()


