# -*- coding: utf-8 -*-
# @Time    : 2020/8/8 16:47
# @Author  : runze.wang

import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--train_folder', default='/data/rz/data/spine/spine_img/train_folder', type=str)
parser.add_argument('--validation_folder', default='/data/rz/data/spine/spine_img/validation_folder', type=str)
parser.add_argument('--is_shuffled', default='1', type=int, help='Needed to shuffle')
parser.add_argument('--train_filename', default='./spine/train_flist', type=str,help='The output filename.')
parser.add_argument('--validation_filename', default='./spine/validation_flist', type=str,help='The output filename.')
if __name__ == "__main__":

    args = parser.parse_args()

    training_file_names = []
    validation_file_names = []

    train_folder_list = os.listdir(args.train_folder)
    for train_name in train_folder_list:
        training_file_names.append(os.path.join(args.train_folder, train_name))

    validation_folder_list = os.listdir(args.validation_folder)
    for validation in validation_folder_list:
        validation_file_names.append(os.path.join(args.validation_folder, validation))

    # print all file paths
    for i in training_file_names:
        print(i)
    for i in validation_file_names:
        print(i)

    # This would print all the files and directories

    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(training_file_names)
        shuffle(validation_file_names)

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    if not os.path.exists(args.validation_filename):
        os.mknod(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)
