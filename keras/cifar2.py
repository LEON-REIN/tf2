# @.@ coding: utf-8 ^-^
# @Author   : Leon Rein
# @Time     : 2020-05-20  ~  20:14 
# @File     : cifar2.py
# @Software : PyCharm
# @Notice   : It's a WINDOWS version!
#             训练一个模型来对飞机airplane和机动车automobile两种图片进行分类
#             ipython 编码问题: 执行 `!chcp 65001` 可以添加至 Setting-Console-Python Console-Starting Script


import tensorflow as tf
from tensorflow.keras import layers, models
import datetime
import os
from matplotlib import pyplot as plt
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # Warnings or Errors ONLY  

'''
Pipeline
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
# map: to accelerate; shuffle: buffer_size need to be large enough; batch: to combine some data into a single batch;
# prefetch: to improves latency and throughput, using additional memory
ds_train = tf.data.Dataset.list_files(r"datasets\cifar2_datasets\train\*\*.jpg") \
    .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=2000) \
    .batch(BATCH_SIZE) \
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

# Display what is in 1 batch while BATCH_SIZE = 100
# for x, y in ds_train.take(1):
#     print(x.shape, y.shape)  # (100, 32, 32, 3) (100,)
# print(list(ds_train.take(1).as_numpy_iterator())[0][1])

'''
3. Keras Modeling (Using  Functional APIs)
'''

tf.keras.backend.clear_session()  # Clear sessions

inputs = layers.Input(shape=(32, 32, 3))  # The shape of input tensor
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.binary_crossentropy,  # Or 'binary_crossentropy'
    metrics=["accuracy"]
)

'''
4. Train the Model
'''

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Time stamp
logdir = os.path.join('datasets', 'cifar2_autograph', stamp)  # path to log
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

history_fit = model.fit(ds_train,
                        epochs=10,
                        validation_data=ds_test,
                        callbacks=[tensorboard_callback],  # Called on batch ends or certain timepoint
                        workers=4,
                        use_multiprocessing=True
                        )

'''
5. Evaluate the Model
'''

# Execute "!tensorboard --logdir datasets\cifar2_autograph" in Ipython
# And open your browser to use Tensorboard
dfhistory = pd.DataFrame(history_fit.history)
dfhistory.index = range(1, len(dfhistory) + 1)
dfhistory.index.name = 'epoch'
print('*************************\n', dfhistory)


# Plot the trend of loss or metrics
def plot_metric(history, indicator):
    train_metrics = history.history[indicator]
    val_metrics = history.history['val_' + indicator]
    epochs = range(1, len(train_metrics) + 1)  # from 1 to epoch+1
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + indicator)
    plt.xlabel("Epochs")
    plt.ylabel(indicator)
    plt.legend(["train_" + indicator, 'val_' + indicator])
    plt.show()


plot_metric(history_fit, "loss")
plot_metric(history_fit, "accuracy")
evaluation = model.evaluate(ds_test, workers=4)
print("Loss: {loss}  Accuracy: {acc}".format(loss=evaluation[0], acc=evaluation[1]))

'''
6. Use the Model
'''

answer = model.predict(ds_test)

# for x, y in ds_test.take(1):
#     print(model.predict_on_batch(x[0:20]))  # Or model.predict(ds_test)

'''
7. Save the Model
'''

# # # Save model weights only
# # model.save_weights(r'cifar2\tf_model_weights.ckpt', save_format="tf")
#
# # Save the whole model
# model.save(r'cifar2\tf_model_savedmodel', save_format="tf")

# # Restore the model
# model_loaded = tf.keras.models.load_model(r'cifar2\tf_model_savedmodel')
# evaluation = model.evaluate(ds_test, workers=4)
# print("Loss: {loss}  Accuracy: {acc}".format(loss=evaluation[0], acc=evaluation[1]))
