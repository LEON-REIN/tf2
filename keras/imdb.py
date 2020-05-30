# @.@ coding:utf-8 ^-^
# @Author   : Leon Rein
# @Time     : 2020-05-30  ~  15:04 
# @File     : imdb.py
# @Software : PyCharm
# @Notice   : It's a WINDOWS version!
#


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, preprocessing, optimizers, losses, metrics
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re, string

'''
Pipeline
1. Load datasets-sets && 2. Data Processing 
'''

train_data_path = "./datasets/IMDB/imdb_data/train.csv"
test_data_path = "./datasets/IMDB/imdb_data/test.csv"
# If you read data through pandas: "pd.read_csv(test_data_path, delimiter="\t", header=None)"

MAX_WORDS = 10000  # Consider only the 10000 words with the highest frequency
MAX_LEN = 200  # Maximum 200 words per sample.
BATCH_SIZE = 20

""" Pipeline Here"""


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
    # A string include all punctuations which has been escaped by re.
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
# print(vectorize_layer.get_vocabulary()[0:100])


'''
3. Keras Modeling
'''

'''
4. Train the Model
'''

'''
5. Evaluate the Model
'''

'''
6. Use the Model
'''

'''
7. Save the Model
'''
