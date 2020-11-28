# @.@ coding:utf-8 ^-^
# @Author   : Leon Rein
# @Time     : 2020-05-30  ~  15:04 
# @File     : imdb.py
# @Software : PyCharm
# @Notice   : It's a WINDOWS version!
#


import re
import string
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

'''
Pipeline
1. Load datasets-sets && 2. Data Processing 
'''

train_data_path = "./datasets/IMDB/imdb_data/train.csv"
test_data_path = "./datasets/IMDB/imdb_data/test.csv"
# If you read data through pandas: "pd.read_csv(test_data_path, delimiter="\t", header=None)"

# OUTPUT of Text Vectors in each batch is tf.Tensor: shape=(20, 200)
MAX_WORDS = 10000  # Consider only the 10000 words with the highest frequency
MAX_LEN = 200  # Maximum 200 words per sample.
BATCH_SIZE = 20

"""Pipeline Here"""


def split_line(line):
    arr = tf.strings.split(line, "\t")
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]), tf.int32), axis=0)
    text = tf.expand_dims(arr[1], axis=0)  # reshape it from () to (1,)
    return text, label


# tf.data.TextLineDataset read csv with the default delimiter '\t'.
ds_train_raw = tf.data.TextLineDataset(filenames=[train_data_path]) \
    .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

ds_test_raw = tf.data.TextLineDataset(filenames=[test_data_path]) \
    .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

'''Building a Dictionary'''


def clean_text(text):
    # A string include all punctuations, except for it"'", which has been escaped by re.
    # Use '\\' for escape of metacharacters of re.
    escaped_punctuation = re.escape(string.punctuation.replace("'", ""))
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    cleaned_punctuation = tf.strings.regex_replace(stripped_html,
                                                   '[%s]' % escaped_punctuation, ' ')
    return cleaned_punctuation


vectorize_layer = TextVectorization(
    # Use the 'lower_and_strip_punctuation' provided by default if data_raw doesn't contain '<br />'
    standardize=clean_text,  # Using custom Callables for text splitting and normalization
    split='whitespace',
    max_tokens=MAX_WORDS - 1,  # There's a place for a placeholder
    output_mode='int',
    output_sequence_length=MAX_LEN)

# list(ds_text.as_numpy_iterator())[0].shape = (20, 1); get the 1st batch's text.
ds_text = ds_train_raw.map(lambda text, label: text)
vectorize_layer.adapt(ds_text)
# print(vectorize_layer.get_vocabulary()[0:100])  # len(This Dictionary) = MAX_WORDS - 2

'''Translate Text into  Text Vectors'''

# If you can not get how the codes followed works. Try:
# dataset = tf.data.Dataset.from_tensor_slices(
#     (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.random.uniform(size=(5, 2))))
# sss = ss.map(lambda a, b:(a+1,b-3))  # different from function map()
# print(list(sss))
# list(ds_test)[0][0]: shape=(20, 200), dtype=int64
ds_train = ds_train_raw.map(lambda text, label: (vectorize_layer(text), label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test_raw.map(lambda text, label: (vectorize_layer(text), label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)

'''
3. Keras Modeling(Inherit from base class--Model, to build custom model)
'''

tf.keras.backend.clear_session()


class CnnModel(models.Model):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.embedding = None
        self.conv_1 = None
        self.pool_1 = None
        self.conv_2 = None
        self.pool_2 = None
        self.flatten = None
        self.dense = None

    def build(self, input_shape):
        # Embedding always comes first.
        # Input shape is (None, MAX_LEN).
        # MAX_WORDS is the size of the dictionary, in which every single word will be converted into vector: (1, 7).
        # Its OUTPUT shape is (None, MAX_LEN, 7). "None" is for BATCH_SIZE dimension.
        self.embedding = layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN)
        self.conv_1 = layers.Conv1D(16, kernel_size=5, name="conv_1", activation="relu")
        self.pool_1 = layers.MaxPool1D(name="pool_1")
        self.conv_2 = layers.Conv1D(128, kernel_size=2, name="conv_2", activation="relu")
        self.pool_2 = layers.MaxPool1D(name="pool_2")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation="sigmoid")
        super(CnnModel, self).build(input_shape)

    def call(self, x, **kwargs):
        x = self.embedding(x)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def summary(self, **kwargs):
        x_input = layers.Input(shape=MAX_LEN)  # x_input: shape = (None, MAX_LEN)
        output = self.call(x_input)
        model_to_show = tf.keras.Model(inputs=x_input, outputs=output)
        model_to_show.summary()


model = CnnModel()
model.build(input_shape=(None, MAX_LEN))
model.summary()

'''
4. Train the Model(Custom cycle-training model)
'''


@tf.function
def printbar():
    # tf.timestamp() records how many seconds have passed from 1970.1.1-00:00:00
    today_ts = tf.timestamp() % (24 * 60 * 60)  # The sum of seconds have passed today in London, U.K.

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)  # Hours in Beijing
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return tf.strings.format("0{}", m)
        else:
            return tf.strings.format("{}", m)

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 8 + timestring)


optimizer = optimizers.Nadam()
loss_func = losses.BinaryCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.BinaryAccuracy(name='train_accuracy')

valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.BinaryAccuracy(name='valid_accuracy')


@tf.function
def train_step(model_train, features, labels):
    with tf.GradientTape() as tape:
        predictions = model_train(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model_train.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_train.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)


@tf.function
def valid_step(model_valid, features, labels):
    predictions = model_valid(features, training=False)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


def train_model(model_to_train, ds_to_train, ds_to_valid, epochs):
    for epoch in tf.range(1, epochs + 1):

        for features, labels in ds_to_train:
            train_step(model_to_train, features, labels)

        for features, labels in ds_to_valid:
            valid_step(model_to_train, features, labels)

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'

        if epoch % 1 == 0:  # You can change '1' to change print interval.
            printbar()
            tf.print(tf.strings.format(logs,
                                       (epoch, train_loss.result(), train_metric.result(), valid_loss.result(),
                                        valid_metric.result())))
            tf.print("")

        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()


train_model(model, ds_train, ds_test, epochs=5)

'''
5. Evaluate the Model
'''


def evaluate_model(model_to_evaluate, ds_valid):
    for features, labels in ds_valid:
        valid_step(model_to_evaluate, features, labels)
    logs = 'Valid Loss:{},Valid Accuracy:{}'
    tf.print(tf.strings.format(logs, (valid_loss.result(), valid_metric.result())))

    valid_loss.reset_states()
    train_metric.reset_states()
    valid_metric.reset_states()


evaluate_model(model, ds_test)

'''
6. Use the Model
'''

# # Method.1
steps = len(list(ds_test.unbatch())) // BATCH_SIZE
# You will get warnings if without 'steps'. Tip: 5000 samples in ds_test totally, and now BATCH_SIZE = 50.
answer = model.predict(ds_test, steps=steps)
#
# # Method.2.3.4
# for x_test, _ in ds_test.take(1):
#     print(model(x_test))
#     # The following methods are the same：
#     # print(model.call(x_test))
#     # print(model.predict_on_batch(x_test))

'''
7. Save the Model
'''

# model.save(r'datasets\IMDB\tf_model_savedmodel', save_format="tf")
# print('export saved model.')
#
# model_loaded = tf.keras.models.load_model(r'datasets\IMDB\tf_model_savedmodel')
# model_loaded.predict(ds_test, steps=(len(list(ds_test.unbatch())) // BATCH_SIZE)))

