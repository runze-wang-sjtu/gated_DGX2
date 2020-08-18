# -*- coding: utf-8 -*-
# @Time    : 2020/8/8 16:47
# @Author  : runze.wang

import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--train_folder', default='/data/rz/data/spine_simulation_2D/train', type=str)
parser.add_argument('--val_folder', default='/data/rz/data/spine_simulation_2D/val', type=str)
parser.add_argument('--is_shuffled', default='1', type=int, help='Needed to shuffle')
parser.add_argument('--image_flist_for_train', default='./spine_simulation_2D/image_flist_for_train',
                    type=str,help='The output filename.')
parser.add_argument('--mask_flist_for_train', default='./spine_simulation_2D/mask_flist_for_train',
                    type=str,help='The output filename.')
parser.add_argument('--image_flist_for_val', default='./spine_simulation_2D/image_flist_for_val',
                    type=str,help='The output filename.')
parser.add_argument('--mask_flist_for_val', default='./spine_simulation_2D/mask_flist_for_val',
                    type=str,help='The output filename.')

if __name__ == "__main__":

    args = parser.parse_args()

    train_image_names = []
    train_mask_names = []
    val_image_names = []
    val_mask_names = []

    train_image_list = os.listdir(os.path.join(args.train_folder,'images'))
    for train_image_name in train_image_list:
        train_image_names.append(os.path.join(args.train_folder, 'images',train_image_name))
    # train_mask_list = os.listdir(os.path.join(args.train_folder, 'masks'))
    # for train_mask_name in train_mask_list:
    #     train_mask_names.append(os.path.join(args.train_folder, train_mask_name))

    val_image_list = os.listdir(os.path.join(args.val_folder,'images'))
    for val_image_name in val_image_list:
        val_image_names.append(os.path.join(args.val_folder,'images', val_image_name))
    # val_mask_list = os.listdir(os.path.join(args.val_folder, 'masks'))
    # for val_mask_name in val_mask_list:
    #     val_mask_names.append(os.path.join(args.val_folder, val_mask_name))

    # print all file paths
    # for i in training_file_names:
    #     print(i)
    # for i in validation_file_names:
    #     print(i)

    # This would print all the files and directories

    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(train_image_names)
        shuffle(val_image_names)

    # make output file if not existed
    if not os.path.exists(args.image_flist_for_train):
        os.mknod(args.image_flist_for_train)
    if not os.path.exists(args.mask_flist_for_train):
        os.mknod(args.mask_flist_for_train)

    if not os.path.exists(args.image_flist_for_val):
        os.mknod(args.image_flist_for_val)
    if not os.path.exists(args.mask_flist_for_val):
        os.mknod(args.mask_flist_for_val)

    # write to file
    fo = open(args.image_flist_for_train, "w")
    fo.write("\n".join(train_image_names))
    fo.close()

    for image_name in train_image_names:
        train_mask_names.append(os.path.join(args.train_folder, 'masks',
                                image_name.split('/')[-1].split('.png')[0]+'_implant.png'))
    fo = open(args.mask_flist_for_train, "w")
    fo.write("\n".join(train_mask_names))
    fo.close()

    fo = open(args.image_flist_for_val, "w")
    fo.write("\n".join(val_image_names))
    fo.close()

    for image_name in val_image_names:
        val_mask_names.append(os.path.join(args.val_folder, 'masks',
                             image_name.split('/')[-1].split('.png')[0] + '_implant.png'))
    fo = open(args.mask_flist_for_val, "w")
    fo.write("\n".join(val_mask_names))
    fo.close()

    # print process
    print("done")
