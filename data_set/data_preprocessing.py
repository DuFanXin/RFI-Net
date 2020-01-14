# -*- coding:utf-8 -*-
"""  
#====#====#====#====
# Project Name:     RFI-Net
# File Name:        data_preprocessing
# Using IDE:        PyCharm Community Edition 
# python version:	3.6
# tf version:		1.7
# From HomePage:    https://github.com/DuFanXin/RFI-Net
# Author:           DuFanXin
# Copyright (c) 2019, All Rights Reserved.
#====#====#====#==== 
"""
import numpy as np
import os
import glob
import cv2
import tensorflow as tf
DATA_FILE_PATH = './collected_hfd5.h5'
TRAIN_SET_NAME = 'train_set.tfrecords'
VALIDATION_SET_NAME = 'validation_set.tfrecords'
TEST_SET_NAME = 'test_set.tfrecords'
PREDICT_SET_NAME = 'predict_set.tfrecords'

INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL = 256, 128, 1
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 256, 128, 1
TRAIN_SET_SIZE = 2900   # total 2976
VALIDATION_SET_SIZE = 76


def segment_hdf5_to_required_part():
    import h5py
    with h5py.File('./data_set.h5', 'w') as file_for_write:
        with h5py.File('./collected_hfd5.h5', 'r') as file_for_read:
            for i in range(1, 32):
                tod = file_for_read['03/%02d/tod' % i].value
                rfi_mask = file_for_read['03/%02d/rfi_mask' % i].value
                for j in range(56):
                    file_for_write['%04d/tod' % (56 * (i - 1) + j)] = tod[0:256, 256 * j:256 * (j + 1)]
                    file_for_write['%04d/rfi_mask' % (56 * (i - 1) + j)] = rfi_mask[0:256, 256 * j:256 * (j + 1)]
                print('%s date 03/%d done segmenting %s' % ('*' * 5, i, '*' * 5))
    print('%s the whole file done segmenting %s' % ('*' * 5, '*' * 5))


class DataProcess(object):

    def __init__(self):
        print("initial data preprocessing")

    @staticmethod # pack the data into tfrecords
    def write_img_to_tfrecords(data_file_path=DATA_FILE_PATH):
        import tensorflow as tf
        # from random import shuffle
        import h5py
        # import cv2
        train_set_writer = tf.python_io.TFRecordWriter(os.path.join('../../RFI-Net - 副本/data_set', TRAIN_SET_NAME))
        validation_set_writer = tf.python_io.TFRecordWriter(os.path.join('../../RFI-Net - 副本/data_set', VALIDATION_SET_NAME))
        # test_set_writer = tf.python_io.TFRecordWriter(os.path.join('../data_set', TEST_SET_NAME))

        # all files
        with h5py.File(data_file_path, 'r') as file_for_read:

            # train set
            for index in range(TRAIN_SET_SIZE):
                tod = file_for_read['%04d/tod' % index].value
                rfi_mask = file_for_read['%04d/rfi_mask' % index].value
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[rfi_mask.tobytes()])),
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tod.tobytes()]))
                }))  
                train_set_writer.write(example.SerializeToString())
                if index % 100 == 0:
                    print('Done train_set writing %.2f%%' % (index / TRAIN_SET_SIZE * 100))
            train_set_writer.close()
            print("Done whole train_set writing")

            # validation set
            for index in range(TRAIN_SET_SIZE, TRAIN_SET_SIZE + VALIDATION_SET_SIZE):
                tod = file_for_read['%04d/tod' % index].value
                rfi_mask = file_for_read['%04d/rfi_mask' % index].value
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[rfi_mask.tobytes()])),
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tod.tobytes()]))
                }))
                validation_set_writer.write(example.SerializeToString())
                if index % 10 == 0:
                    print('Done validation_set writing %.2f%%' % ((index - TRAIN_SET_SIZE) / VALIDATION_SET_SIZE * 100))
            validation_set_writer.close()
            print("Done whole validation_set writing")


if __name__ == '__main__':
    mydata = DataProcess()
    # mydata.write_img_to_tfrecords()
