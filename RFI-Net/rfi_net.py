# -*- coding:utf-8 -*-
"""  
#====#====#====#====
# Project Name:     RFI-Net
# File Name:        rfi-net
# Using IDE:        PyCharm Community Edition
# python version:	3.6
# tf version:		1.13
# From HomePage:    https://github.com/DuFanXin/RFI-Net
# Author:           DuFanXin 
# Copyright (c) 2019, All Rights Reserved.
#====#====#====#==== 
"""
import tensorflow as tf
# from tensorflow.contrib.layers import l2_regularizer
import argparse
import os

EPOCH_NUM = 1
TRAIN_BATCH_SIZE = 8
VALIDATION_BATCH_SIZE = 2
TEST_BATCH_SIZE = 1
PREDICT_BATCH_SIZE = 1
TRAIN_SET_SIZE = 2100
TEST_SET_SIZE = 76
EPS = 10e-5
FLAGS = None
CLASS_NUM = 2
TIMES = 2

PREDICT_DIRECTORY = '../data_set/test'
TEST_DIRECTORY = '../data_set/test'
PREDICT_SAVED_DIRECTORY = '../data_set/predictions'
TEST_RESULT_DIRECTORY = '../data_set/test_result/rfi_net_test_result'
CHECK_POINT_PATH = '../data_set/saved_models/train/model.ckpt'
DATA_DIR = '../data_set/'
MODEL_DIR = '../data_set/saved_models'
LOG_DIR = '../data_set/logs'

INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL = 256, 128, 1
OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE, OUTPUT_IMG_CHANNEL = 256, 128, 1

TRAIN_SET_NAME = 'train_set.tfrecords'
VALIDATION_SET_NAME = 'validation_set.tfrecords'
TEST_SET_NAME = 'test_set.h5'


def read_image(file_queue):
    reader = tf.TFRecordReader()
    # reader = tf.data.TFRecordDataset()
    # key, value = reader.read(file_queue)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.float64)
    # print('image ' + str(image))
    image = tf.reshape(image, [INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL])
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image = tf.image.resize_images(image, (IMG_HEIGHT, IMG_WIDE))
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    label = tf.decode_raw(features['label'], tf.uint8)
    # label = tf.cast(label, tf.int64)
    label = tf.reshape(label, [OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE])
    # label = tf.decode_raw(features['image_raw'], tf.uint8)
    # print(label)
    # label = tf.reshape(label, shape=[1, 4])
    return image, label


def read_image_batch(file_queue, batch_size):
    img, label = read_image(file_queue)
    min_after_dequeue = 2000
    capacity = 4000
    # image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
    image_batch, label_batch = tf.train.shuffle_batch(
        tensors=[img, label], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)
    # one_hot_labels = tf.to_float(tf.one_hot(indices=label_batch, depth=CLASS_NUM))
    one_hot_labels = tf.reshape(label_batch, [batch_size, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE])
    return image_batch, one_hot_labels


class RFI_Net:

    def __init__(self, train_set_name=TRAIN_SET_NAME, test_set_name=TEST_SET_NAME, validation_set_name=VALIDATION_SET_NAME,
                 input_img_height=INPUT_IMG_HEIGHT, input_img_wide=INPUT_IMG_WIDE, input_img_channel=INPUT_IMG_CHANNEL,
                 output_img_height=OUTPUT_IMG_HEIGHT, output_img_wide=OUTPUT_IMG_WIDE):
        print('New RFI_Net Network')
        self.input_image = None
        self.input_label = None
        self.cast_image = None
        self.cast_label = None
        self.keep_prob = None
        self.lamb = None
        self.result_expand = None
        self.is_traing = None
        self.loss, self.loss_mean, self.loss_all, self.train_step = [None] * 4
        self.prediction, self.correct_prediction, self.accuracy = [None] * 3
        self.result_conv = {}
        self.result_relu = {}
        self.result_maxpool = {}
        self.result_from_contract_layer = {}
        self.w_0 = None
        self.learning_rate = None
        self.train_set_name = train_set_name
        self.test_set_name = test_set_name
        self.validation_set_name = validation_set_name
        self.input_img_height = input_img_height
        self.input_img_wide = input_img_wide
        self.input_img_channel = input_img_channel
        self.output_img_height = output_img_height
        self.output_img_wide = output_img_wide
    # self.b = {}

    def init_w(self, shape, name=None):
        stddev = tf.sqrt(x=2 / (shape[0] * shape[1] * shape[2] * shape[3]))
        w = tf.get_variable(
            name=name,  # regularizer=l2_regularizer(scale=self.lamb),
            initializer=tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32))
        return w

    @staticmethod
    def init_b(shape, name):
        with tf.name_scope('init_b'):
            return tf.Variable(initial_value=tf.random_normal(shape=shape, dtype=tf.float32), name=name)

    @staticmethod
    def batch_norm(x, is_training, eps=EPS, decay=0.9, affine=True, var_scope_name='BatchNorm2d'):
        from tensorflow.python.training.moving_averages import assign_moving_average

        with tf.variable_scope(var_scope_name):
            params_shape = x.shape[-1:]
            moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False)
            moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer, trainable=False)

            def mean_var_with_update():
                mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
                with tf.control_dependencies([
                    assign_moving_average(moving_mean, mean_this_batch, decay),
                    assign_moving_average(moving_var, variance_this_batch, decay)
                ]):
                    return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

            mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))
            if affine:  # If you want to scale with beta and gamma
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=eps)
            else:
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,  variance_epsilon=eps)
            return normed

    @staticmethod
    def copy_and_merge(result_from_contract_layer, result_from_upsampling):
        result_from_contract_layer_crop = result_from_contract_layer
        return tf.concat(values=[result_from_contract_layer_crop, result_from_upsampling], axis=-1)

    # residual unit for down sampling
    def res_unit_down(self, layer_num, input_data):
        layer_name = 'res_unit_down_%d' % layer_num
        channels_num = input_data.get_shape().as_list()[-1]
        # print(type(channels_num))
        with tf.variable_scope(layer_name):
            # split from the input to short connect
            w_0 = self.init_w(shape=[1, 1, channels_num, 2 * channels_num], name='w_0')
            result_conv_0 = tf.nn.conv2d(
                input=input_data, filter=w_0, strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            split_from_input = self.batch_norm(
                x=result_conv_0, is_training=self.is_traing, var_scope_name='%s_split' % layer_name)

            # conv_1
            # w_1 = self.init_w(shape=[1, 1, channels_num, channels_num // 2], name='w_1')
            # result_conv_1 = tf.nn.conv2d(
            # 	input=input_data, filter=w_1, strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            # normed_batch = self.batch_norm(
            # 	x=result_conv_1, is_training=self.is_traing, var_scope_name='%s_conv_1' % layer_name)
            # result_relu_1 = tf.nn.relu(normed_batch, name='relu')

            # conv_3
            w_2 = self.init_w(shape=[3, 3, channels_num, 2 * channels_num], name='w_2')
            result_conv_2 = tf.nn.conv2d(
                input=input_data, filter=w_2, strides=[1, 1, 1, 1], padding='SAME', name='conv_3')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, var_scope_name='%s_conv_2' % layer_name)
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu')

            # conv_3
            w_3 = self.init_w(shape=[3, 3, 2 * channels_num, 2 * channels_num], name='w_3')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_2, filter=w_3, strides=[1, 1, 1, 1], padding='SAME', name='conv_3')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, var_scope_name='%s_conv_3' % layer_name)
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu')

            # conv_4
            w_4 = self.init_w(shape=[3, 3, 2 * channels_num, 2 * channels_num], name='w_4')
            result_conv_1 = tf.nn.conv2d(
                input=result_relu_2, filter=w_4, strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing, var_scope_name='%s_conv_4' % layer_name)

            # add short skip
            result_add = tf.add(x=normed_batch, y=split_from_input, name='add')
            result_add = self.batch_norm(x=result_add, is_training=self.is_traing, var_scope_name='%s_add' % layer_name)
            result_relu_add = tf.nn.relu(result_add, name='relu')

            return result_relu_add

    # residual unit for down sampling
    def res_unit_up(self, layer_num, input_data):
        layer_name = 'res_unit_up_%d' % layer_num
        channels_num = input_data.get_shape().as_list()[-1]
        with tf.variable_scope(layer_name):
            # split from the input to short connect
            w_0 = self.init_w(shape=[1, 1, channels_num, channels_num // 2], name='w_0')
            result_conv_0 = tf.nn.conv2d(
                input=input_data, filter=w_0, strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            split_from_input = self.batch_norm(
                x=result_conv_0, is_training=self.is_traing, var_scope_name='%s_split' % layer_name)

            # conv_1
            w_1 = self.init_w(shape=[3, 3, channels_num, channels_num // 2], name='w_1')
            result_conv_1 = tf.nn.conv2d(
                input=input_data, filter=w_1, strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch = self.batch_norm(x=result_conv_1, is_training=self.is_traing, var_scope_name='%s_conv_1' % layer_name)
            result_relu_1 = tf.nn.relu(normed_batch, name='relu')

            # conv_2
            w_2 = self.init_w(shape=[3, 3, channels_num // 2, channels_num // 2], name='w_2')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=w_2, strides=[1, 1, 1, 1], padding='SAME', name='conv_3')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, var_scope_name='%s_conv_2' % layer_name)
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu')

            # conv_3
            w_3 = self.init_w(shape=[3, 3, channels_num // 2, channels_num // 2], name='w_3')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_2, filter=w_3, strides=[1, 1, 1, 1], padding='SAME', name='conv_3')
            normed_batch = self.batch_norm(x=result_conv_2, is_training=self.is_traing, var_scope_name='%s_conv_3' % layer_name)
            # result_relu_2 = tf.nn.relu(features=normed_batch, name='relu')

            # conv_1
            # w_4 = self.init_w(shape=[1, 1, channels_num / 2, 2 * channels_num], name='w_4')
            # result_conv_1 = tf.nn.conv2d(
            # 	input=result_relu_2, filter=w_4, strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            # normed_batch = self.batch_norm(
            # 	x=result_conv_1, is_training=self.is_traing, var_scope_name='%s_conv_1' % layer_name)

            # add short skip
            result_add = tf.add(x=normed_batch, y=split_from_input, name='add')
            result_add = self.batch_norm(x=result_add, is_training=self.is_traing, var_scope_name='%s_add' % layer_name)
            result_relu_add = tf.nn.relu(result_add, name='relu')
            return result_relu_add

    def up_sample(self, layer_num, input_data):
        batch_size, height, wide, channels_num = input_data.get_shape().as_list()
        w_upsample = self.init_w(shape=[2, 2, channels_num // 2, channels_num], name='w_upsample')
        # self.b[11] = self.init_b(shape=[512], name='b_11')
        result_up = tf.nn.conv2d_transpose(
            value=input_data, filter=w_upsample,
            output_shape=[batch_size, height * 2, wide * 2, channels_num // 2],
            strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
        normed_batch = self.batch_norm(x=result_up, is_training=self.is_traing, var_scope_name='layer_%d_conv_up' % layer_num)
        result_relu_3 = tf.nn.relu(features=normed_batch, name='relu')
        return result_relu_3

    def set_up_net(self, batch_size):
        # input
        with tf.name_scope('input'):
            # learning_rate = tf.train.exponential_decay()
            self.input_image = tf.placeholder(
                dtype=tf.float32, shape=[batch_size, self.input_img_height, self.input_img_wide, self.input_img_channel], name='input_images'
            )

            self.input_label = tf.placeholder(
                dtype=tf.int32, shape=[batch_size, self.output_img_height, self.output_img_wide], name='input_labels'
            )
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
            self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')
            self.is_traing = tf.placeholder(dtype=tf.bool, name='is_traing')
            normed_batch = self.batch_norm(x=self.input_image, is_training=self.is_traing, var_scope_name='input')

        # layer 1
        with tf.name_scope('layer_1'), tf.variable_scope('layer_1'):

            # expand the channel of input_data to required num
            w_expand = self.init_w(shape=[3, 3, INPUT_IMG_CHANNEL, 32], name='w_expand')
            result_conv = tf.nn.conv2d(
                input=normed_batch, filter=w_expand, strides=[1, 1, 1, 1], padding='SAME', name='conv')
            normed_batch = self.batch_norm(x=result_conv, is_training=self.is_traing, var_scope_name='layer_1_expand')
            result_relu = tf.nn.relu(features=normed_batch, name='relu')

            # res_unit_down
            result_res_unit_down = self.res_unit_down(layer_num=1, input_data=result_relu)

            # save the temporary results in this layer for the up_sample path
            self.result_from_contract_layer[1] = result_res_unit_down

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_res_unit_down, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, rate=1 - self.keep_prob)

        # layer 2
        with tf.name_scope('layer_2'), tf.variable_scope('layer_2'):
            # res_unit_down
            result_res_unit_down = self.res_unit_down(layer_num=2, input_data=result_dropout)

            # save the temporary results in this layer for the up_sample path
            self.result_from_contract_layer[2] = result_res_unit_down

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_res_unit_down, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, rate=1-self.keep_prob)

        # layer 3
        with tf.name_scope('layer_3'), tf.variable_scope('layer_3'):
            # res_unit_down
            result_res_unit_down = self.res_unit_down(layer_num=3, input_data=result_dropout)

            # save the temporary results in this layer for the up_sample path
            self.result_from_contract_layer[3] = result_res_unit_down

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_res_unit_down, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, rate=1 - self.keep_prob)

        # layer 4
        with tf.name_scope('layer_4'), tf.variable_scope('layer_4'):
            # res_unit_down
            result_res_unit_down = self.res_unit_down(layer_num=4, input_data=result_dropout)

            # save the temporary results in this layer for the up_sample path
            self.result_from_contract_layer[4] = result_res_unit_down

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_res_unit_down, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, rate=1 - self.keep_prob)

        # layer 5 (bottom 16 * 8 * 1024)
        with tf.name_scope('layer_5'), tf.variable_scope('layer_5'):
            # res_unit_down
            result_res_unit_down = self.res_unit_down(layer_num=5, input_data=result_dropout)

            # up sample
            result_relu_3 = self.up_sample(layer_num=5, input_data=result_res_unit_down)

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, rate=1 - self.keep_prob)
            # print(result_dropout.shape)

        # layer 6
        with tf.name_scope('layer_6'), tf.variable_scope('layer_6'):
            # copy and merge
            result_merge = self.copy_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[4], result_from_upsampling=result_dropout)
            result_merge_normed = self.batch_norm(x=result_merge, is_training=self.is_traing, var_scope_name='layer_6_merge')
            # print(result_merge)

            # res_unit_up
            result_res_unit_up = self.res_unit_up(layer_num=6, input_data=result_merge_normed)

            # up sample
            result_relu_3 = self.up_sample(layer_num=5, input_data=result_res_unit_up)

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, rate=1 - self.keep_prob)

        # layer 7
        with tf.name_scope('layer_7'), tf.variable_scope('layer_7'):
            # copy and merge
            result_merge = self.copy_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[3], result_from_upsampling=result_dropout)
            result_merge_normed = self.batch_norm(x=result_merge, is_training=self.is_traing, var_scope_name='layer_7_merge')

            # res_unit_up
            result_res_unit_up = self.res_unit_up(layer_num=7, input_data=result_merge_normed)

            # up sample
            result_relu_3 = self.up_sample(layer_num=5, input_data=result_res_unit_up)

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, rate=1 - self.keep_prob)

        # layer 8
        with tf.name_scope('layer_8'), tf.variable_scope('layer_8'):
            # copy and merge
            result_merge = self.copy_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[2], result_from_upsampling=result_dropout)
            result_merge_normed = self.batch_norm(x=result_merge, is_training=self.is_traing, var_scope_name='layer_8_merge')

            # res_unit_up
            result_res_unit_up = self.res_unit_up(layer_num=8, input_data=result_merge_normed)

            # up sample
            result_relu_3 = self.up_sample(layer_num=5, input_data=result_res_unit_up)

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, rate=1 - self.keep_prob)

        # layer 9
        with tf.name_scope('layer_9'), tf.variable_scope('layer_9'):
            # copy and merge
            result_merge = self.copy_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[1], result_from_upsampling=result_dropout)
            result_merge_normed = self.batch_norm(x=result_merge, is_training=self.is_traing, var_scope_name='layer_9_merge')

            # res_unit_up
            result_res_unit_up = self.res_unit_up(layer_num=9, input_data=result_merge_normed)

            # convolution to [self.batch_size, OUTPIT_IMG_WIDE, OUTPUT_IMG_HEIGHT, CLASS_NUM
            w = self.init_w(shape=[1, 1, 64, CLASS_NUM], name='w')
            # self.b[23] = self.init_b(shape=[CLASS_NUM], name='b_11')
            result_conv_3 = tf.nn.conv2d(
                input=result_res_unit_up, filter=w,
                strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
            normed_batch = self.batch_norm(x=result_conv_3, is_training=self.is_traing, var_scope_name='layer_9_conv_3')

            # softmax
            self.prediction = normed_batch
            # print(self.prediction.shape)
        # Mean Squared Error
        # self.prediction = tf.argmax(input=normed_batch, axis=-1, output_type=tf.int32)

        # loss(chose one in two loss function)
        with tf.name_scope('loss'):

            # softmax
            self.loss = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
            self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

            # Mean Squared Error
            # self.loss_mean = tf.reduce_mean(tf.square(x=tf.to_float(self.input_label - self.prediction)), name='reduce_mean')

            tf.add_to_collection(name=tf.GraphKeys.REGULARIZATION_LOSSES, value=self.loss_mean)
            self.loss_all = tf.reduce_sum(input_tensor=tf.get_collection(key=tf.GraphKeys.REGULARIZATION_LOSSES))

        # accuracy(chose one in two)
        with tf.name_scope('accuracy'):
            # softmax
            self.correct_prediction = \
                tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32), self.input_label)
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)

        # Mean Squared Error
        # self.correct_prediction = tf.equal(self.prediction, self.input_label)
        # self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
        # self.accuracy = tf.reduce_mean(self.correct_prediction)

        # Gradient Descent
        with tf.name_scope('Gradient_Descent'):
            global_step = tf.Variable(0, trainable=False)
            decay_steps = TRAIN_SET_SIZE * EPOCH_NUM / TRAIN_BATCH_SIZE / 10
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=1e-4, global_step=global_step, decay_steps=decay_steps, decay_rate=0.90, staircase=True)
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_all, global_step=global_step)

    def train(self, train_batch_size=TRAIN_BATCH_SIZE, train_file_path=None, log_path=None, model_file_path=None,
              model_name="model.ckpt"):

        if train_file_path is None:
            train_file_path = os.path.join(DATA_DIR, TRAIN_SET_NAME)

        if log_path is None:
            log_path = os.path.join(LOG_DIR, "rfi_net")

        if model_file_path is None:
            model_file_path = os.path.join(MODEL_DIR, model_name)

        self.set_up_net(batch_size=train_batch_size)

        train_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=EPOCH_NUM, shuffle=True)
        # train_image_filename_queue = tf.data.Dataset.from_tensor_slices(
        #     tf.train.match_filenames_once(train_file_path))\
        #     .shuffle(tf.shape(tf.train.match_filenames_once(train_file_path), out_type=tf.int64)[0]).repeat(EPOCH_NUM)
        # model_file_path = os.path.join(FLAGS.model_dir, "model.ckpt")
        train_images, train_labels = read_image_batch(train_image_filename_queue, train_batch_size)
        tf.summary.scalar("loss", self.loss_mean)
        tf.summary.scalar('accuracy', self.accuracy)
        merged_summary = tf.summary.merge_all()
        # all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  # start a session
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            summary_writer = tf.summary.FileWriter(log_path, sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                epoch = 1
                while not coord.should_stop():
                    # Run training steps or whatever
                    # print('epoch ' + str(epoch))
                    example, label = sess.run([train_images, train_labels])  # fetch image and label in a session
                    # print(label)
                    lo, acc, summary_str = sess.run(
                        [self.loss_mean, self.accuracy, merged_summary],
                        feed_dict={
                            self.input_image: example, self.input_label: label, self.keep_prob: 1.0,
                            self.lamb: 0.004, self.is_traing: True}
                    )
                    summary_writer.add_summary(summary_str, epoch)
                    print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
                    if epoch % 10 == 0:
                        print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
                    sess.run(
                        [self.train_step],
                        feed_dict={
                            self.input_image: example, self.input_label: label, self.keep_prob: 1.0,
                            self.lamb: 0.004, self.is_traing: True}
                    )
                    epoch += 1
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                # all_parameters_saver.save(sess=sess, save_path=model_file_path)
                coord.request_stop()
            # coord.request_stop()
            coord.join(threads)
        print("Done training")

    def validate(self, validation_file_path=None, model_file_path=None, model_name="model.ckpt"):
        import time

        self.set_up_net(batch_size=VALIDATION_BATCH_SIZE)

        if model_file_path is None:
            model_file_path = os.path.join(MODEL_DIR, model_name)

        if validation_file_path is None:
            validation_file_path = os.path.join(DATA_DIR, VALIDATION_SET_NAME)

        validation_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(validation_file_path), num_epochs=1, shuffle=True)
        # model_file_path = '../data_set/saved_models/3rd/model.ckpt'  # CHECK_POINT_PATH
        validation_images, validation_labels = read_image_batch(validation_image_filename_queue, VALIDATION_BATCH_SIZE)
        # tf.summary.scalar("loss", self.loss_mean)
        # tf.summary.scalar('accuracy', self.accuracy)
        # merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  # 开始一个会话
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            # tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            all_parameters_saver.restore(sess=sess, save_path=model_file_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            try:
                epoch = 1
                while not coord.should_stop():
                    # Run training steps or whatever
                    # print('epoch ' + str(epoch))
                    example, label = sess.run([validation_images, validation_labels])  # 在会话中取出image和label
                    # print(label)
                    lo, acc = sess.run(
                        [self.loss_mean, self.accuracy],
                        feed_dict={
                            self.input_image: example, self.input_label: label, self.keep_prob: 1.0,
                            self.lamb: 0.004, self.is_traing: False}
                    )
                    # summary_writer.add_summary(summary_str, epoch)
                    # print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
                    if epoch % 1 == 0:
                        print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
                    epoch += 1
            except tf.errors.OutOfRangeError:
                used_time = time.time() - start_time
                print('Done validating -- epoch limit reached, use %ds, average %.2fs/pic' % (used_time, used_time / 76))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            # coord.request_stop()
            coord.join(threads)
        print('Done validating')


    def test(self, test_set_size=TEST_SET_SIZE, test_file_path=None, test_result_path=None, model_file_path=None, model_name="model.ckpt"):
        import cv2
        import time
        import numpy as np
        import h5py as h5

        # test_file_path = glob.glob(os.path.join(TEST_DIRECTORY, '*.tif'))
        # print('Tatol %d images to test' % len(test_file_path))
        # ckpt_path = '../data_set/saved_models/3rd/model.ckpt'  # CHECK_POINT_PATH
        if model_file_path is None:
            model_file_path = os.path.join(MODEL_DIR, model_name)

        if test_file_path is None:
            test_file_path = os.path.join(DATA_DIR, TEST_SET_NAME)

        if test_result_path is None:
            test_result_path = os.path.join(TEST_RESULT_DIRECTORY, 'Score_temp.h5')

        if not os.path.lexists(test_result_path):
            os.mkdir(test_result_path)

        file_to_read = h5.File(test_file_path, 'r')
        file_to_write = h5.File(test_result_path, 'w')
        all_parameters_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            all_parameters_saver.restore(sess=sess, save_path=model_file_path)
            start_time = time.time()
            for index in range(test_set_size):
                tod = np.reshape(
                    a=file_to_read['%02d/tod' % index].value,
                    newshape=(1, self.input_img_height, self.input_img_wide, self.input_img_channel))

                prediction = sess.run(
                    self.prediction,
                    feed_dict={self.input_image: tod, self.keep_prob: 1.0, self.lamb: 0.004, self.is_traing: False})

                predict_softmax = sess.run(tf.nn.softmax(logits=prediction))
                file_to_write['%02d/class_0' % index] = predict_softmax[0][:, :, 0]
                file_to_write['%02d/class_1' % index] = predict_softmax[0][:, :, 1]

                predict_image = sess.run(tf.cast(x=tf.argmax(input=prediction, axis=-1), dtype=tf.uint8))
                cv2.imwrite(os.path.join(TEST_RESULT_DIRECTORY, '%d_temp.jpg' % index), predict_image[0] * 255)  # * 255
                file_to_write['%02d/predict' % index] = predict_image[0]
                file_to_write['%02d/ground_truth' % index] = file_to_read['%02d/rfi_mask' % index].value
                if index % 10 == 0:
                    print('Done testing %.2f%%' % (index / TEST_SET_SIZE * 100))
            file_to_read.close()
            file_to_write.close()
        used_time = time.time() - start_time
        print('Done testing, test result in floder test_saved, use %ds, average %.2fs/pic' % (used_time, used_time / TEST_SET_SIZE))

    def predict(self, predict_path=PREDICT_DIRECTORY, predict_saved_path=PREDICT_SAVED_DIRECTORY, model_file_path=None, model_name="model.ckpt"):
        import cv2
        import glob
        import numpy as np

        predict_file_path = glob.glob(os.path.join(predict_path, '*.tif'))
        print('Tatol %d images to predict' % len(predict_file_path))
        # ckpt_path = '../data_set/saved_models/1st/model.ckpt'  # CHECK_POINT_PATH

        if model_file_path is None:
            model_file_path = os.path.join(MODEL_DIR, model_name)

        if not os.path.lexists(predict_saved_path):
            os.mkdir(predict_saved_path)
        all_parameters_saver = tf.train.Saver()

        with tf.Session() as sess:  # start a session
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            # tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            all_parameters_saver.restore(sess=sess, save_path=model_file_path)
            for index, image_path in enumerate(predict_file_path):
                print(image_path)
                # image = cv2.imread(image_path, flags=0)
                image = np.reshape(
                    a=cv2.imread(image_path, flags=0),
                    newshape=(1, self.input_img_wide, self.input_img_height, self.input_img_channel))
                predict_image = sess.run(
                    tf.argmax(input=self.prediction, axis=3),
                    feed_dict={
                        self.input_image: image, self.keep_prob: 1.0, self.lamb: 0.004, self.is_traing: False
                    }
                )
                cv2.imwrite(os.path.join(predict_saved_path, image_path[image_path.rindex('/') + 1:]), predict_image[0] * 255)  # * 255

        print('Done prediction')

    def test_time(self, test_batch_size, test_size):    # test wtriting time
        import time
        import numpy as np
        import h5py as h5
        self.set_up_net(batch_size=test_batch_size)

        file_to_read = h5.File(os.path.join('../data_set', 'test_set.h5'), 'r')
        file_to_write = h5.File(os.path.join(TEST_RESULT_DIRECTORY, 'Score_temp.h5'), 'w')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=(5 / 16))
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            tod = np.ones(shape=[1, self.input_img_height, self.input_img_wide, self.input_img_channel], dtype=np.float64)
            start_time = time.time()
            for index in range(test_size):

                prediction = sess.run(
                    self.prediction,
                    feed_dict={self.input_image: tod, self.keep_prob: 1.0, self.lamb: 0.004, self.is_traing: False})

                # predict_softmax = sess.run(tf.nn.softmax(logits=prediction))
                # file_to_write['%02d/class_0' % index] = predict_softmax[0][:, :, 0]
                # file_to_write['%02d/class_1' % index] = predict_softmax[0][:, :, 1]

                predict_image = sess.run(tf.cast(x=tf.argmax(input=prediction, axis=-1), dtype=tf.uint8))
                # cv2.imwrite(os.path.join(TEST_RESULT_DIRECTORY, '%d.jpg' % index), predict_image[0] * 255)  # * 255
                file_to_write['%02d/predict' % index] = predict_image[0]
                # file_to_write['%02d/ground_truth' % index] = file_to_read['%02d/rfi_mask' % index].value
                if index % 10 == 0:
                    print('Done testing %.2f%%' % (index / test_size * 100))
            used_time = time.time() - start_time
            print('Done testing time, use %.4fs, total %d pics, average %.4fs/pic' % (used_time, test_size, used_time / test_size))


def main():
    net = RFI_Net()
    # net.set_up_net(TRAIN_BATCH_SIZE)
    net.train()
    # net.set_up_net(VALIDATION_BATCH_SIZE)
    # net.validate()
    # net.set_up_net(TEST_BATCH_SIZE)
    # net.test_time(test_batch_size=1, test_size=76, height=256, wide=128)
    # net.set_up_net(PREDICT_BATCH_SIZE)
    # net.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data path
    parser.add_argument(
        '--data_dir', type=str, default=DATA_DIR,
        help='Directory for storing input data_set')

    # model saved into
    parser.add_argument(
        '--model_dir', type=str, default=MODEL_DIR,
        help='output model path')

    # log saved into
    parser.add_argument(
        '--log_dir', type=str, default=LOG_DIR,
        help='TensorBoard log path')

    FLAGS, _ = parser.parse_known_args()

    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    # write_img_to_tfrecords()
    # read_check_tfrecords()
    main()
