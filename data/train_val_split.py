# -*- coding: utf-8 -*-
# @Time    : 2020/8/6 10:46
# @Author  : runze.wang

import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/home/gdp/data/spine_simulation_2D', type=str,help='The folder path')
parser.add_argument('--train_folder', default='/home/gdp/data/spine/spine_img/train_folder', type=str)
parser.add_argument('--validation_folder', default='/home/gdp/data/spine/spine_img/validation_folder', type=str)

def copy_file(source_dir_list, target_dir):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for dir in source_dir_list:
        shutil.copy(dir, target_dir)

if __name__ == "__main__":
    args = parser.parse_args()

    dirs = os.listdir(args.folder_path)

    train_name_list = []
    validation_name_list = []

    for i in range(len(dirs)):
        if dirs[i].split('.')[-1] == 'png':
            if i <= int(0.8*len(dirs)):
                train_name_list.append(os.path.join(args.folder_path, dirs[i]))
            else:
                validation_name_list.append(os.path.join(args.folder_path, dirs[i]))
    copy_file(train_name_list, args.train_folder)
    copy_file(validation_name_list, args.validation_folder)



