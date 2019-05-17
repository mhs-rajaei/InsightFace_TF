# -*-coding: utf-8 -*-
"""
    https://github.com/PanJinquan/Face_Detection_Recognition
    @Project: IntelligentManufacture
    @File   : file_processing.py
    @Author : panjq
    @ E-mail: pan_jinquan@163.com
    @Date   : 2019-02-14 15:08:19
"""
import glob
import os
import shutil
import numpy as np
import pandas as pd


def write_data(filename, content_list,mode='w'):
    """Save the data of list[list[]] to the txt file
    :param filename: filename
    :param content_list: data to be saved, type->list
    :param mode: read-write mode: 'w' or 'a'
    :return: void
    """
    with open(filename, mode=mode, encoding='utf-8') as f:
        for line_list in content_list:
            # Convert list to string
            line=" ".join('%s' % id for id in line_list)
            f.write(line+"\n")


def write_list_data(filename, list_data,mode='w'):
    """Save list[] data to txt file, each element branch
    :param filename: filename
    :param list_data: data to be saved, type->list
    :param mode: read-write mode: 'w' or 'a'
    :return: void
    """
    with open(filename, mode=mode, encoding='utf-8') as f:
        for line in list_data:
            # Convert list to string
            f.write(str(line)+"\n")


def read_data(filename, split=" ", convertNum=True):
    """
    Read txt data function
    :param filename: filename
    :param split : separator
    :param convertNum : Whether to convert the string in the list to a number of type int/float
    :return: txt data list
    There are three functions in Python that remove the head and tail characters and whitespace characters, which are:
    Strip: Used to remove head and tail characters, whitespace characters (including \n, \r, \t, ' ', ie: line feed, carriage return, tab, space)
    Lstrip: used to remove the beginning character, white space (including \n, \r, \t, ' ', ie: line feed, carriage return, tab, space)
    Rstrip: used to remove trailing characters, whitespace characters (including \n, \r, \t, ' ', ie: line feed, carriage return, tab, space)
    Note: These functions will only remove the first and last characters, and the middle will not be deleted.
    """
    with open(filename, mode="r",encoding='utf-8') as f:
        content_list = f.readlines()
        if split is None:
            content_list = [content.rstrip() for content in content_list]
            return content_list
        else:
            content_list = [content.rstrip().split(split) for content in content_list]
        if convertNum:
            for i,line in enumerate(content_list):
                line_data=[]
                for l in line:
                    if is_int(l):  # isdigit() method to detect whether a string consists of only numbers, only integers
                        line_data.append(int(l))
                    elif is_float(l):  # Determine if it is a decimal
                        line_data.append(float(l))
                    else:
                        line_data.append(l)
                content_list[i]=line_data
    return content_list


def is_int(str):
    # Determine whether it is an integer
    try:
        x = int(str)
        return isinstance(x, int)
    except ValueError:
        return False


def is_float(str):
    # Determine whether it is an integer and a decimal
    try:
        x = float(str)
        return isinstance(x, float)
    except ValueError:
        return False


def list2str(content_list):
    content_str_list=[]
    for line_list in content_list:
        line_str = " ".join('%s' % id for id in line_list)
        content_str_list.append(line_str)
    return content_str_list


def get_images_list(image_dir, postfix=['*.jpg'], basename=False):
    """
    Get a list of files
    :param image_dir: image file directory
    :param postfix: suffix name, but multiple, such as ['*.jpg', '*.png']
    :param basename: The returned list is the file name (True), or the full path of the file (False)
    :return:
    """
    images_list=[]
    for format in postfix:
        image_format=os.path.join(image_dir,format)
        image_list=glob.glob(image_format)
        if not image_list==[]:
            images_list+=image_list
    images_list=sorted(images_list)
    if basename:
        images_list=get_basename(images_list)
    return images_list

def get_basename(file_list):
    dest_list=[]
    for file_path in file_list:
        basename=os.path.basename(file_path)
        dest_list.append(basename)
    return dest_list


def copyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile) #分离 filename and path
        if not os.path.exists(fpath):
            os.makedirs(fpath) #Create path
        shutil.copyfile(srcfile,dstfile) #copy file
        # print("copy %s -> %s"%( srcfile,dstfile))


def merge_list(data1, data2):
    """
    Combine two lists
    : param data1:
    :param data2:
    :return: returns the merged list
    """
    if not len(data1) == len(data2):
        return
    all_data = []
    for d1, d2 in zip(data1, data2):
        all_data.append(d1 + d2)
    return all_data


def split_list(data, split_index=1):
    """
    Divide data into two parts
    :param data: list
    :param split_index: the location of the split
    :return:
    """
    data1 = []
    data2 = []
    for d in data:
        d1 = d[0:split_index]
        d2 = d[split_index:]
        data1.append(d1)
        data2.append(d2)
    return data1, data2


def getFilePathList(file_dir):
    """
    Get all text paths in the file_dir directory, including subdirectory files
    :param rootDir:
    :return:
    """
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list


def get_files_list(file_dir, postfix=None):
    """
    Obtain all file lists with postfix name postfix in the file_dir directory, including subdirectories
    : param file_dir:
    :param postfix: ['*.jpg','*.png'], postfix=None means all files
    :return:
    """
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix is None:
        file_list = filePath_list
    else:
        postfix = [p.split('.')[-1] for p in postfix]
        for file in filePath_list:
            basename = os.path.basename(file) # Get the file name under the path
            postfix_name = basename.split('.')[-1]
            if postfix_name in postfix:
                file_list.append(file)
    file_list.sort()
    return file_list


def gen_files_labels(files_dir,postfix=None):
    """
    Get all file paths under the files_dir path, as well as labels, where labels are represented by child file names
    In the files_dir directory, a folder of the same category is placed in a folder, and its labels are the names of the files.
    : param files_dir:
    :postfix suffix
    :return:filePath_list The path to all files, labels corresponding to label_list
    """
    # filePath_list = getFilePathList(files_dir)
    filePath_list=get_files_list(files_dir, postfix=postfix)
    print("files nums:{}".format(len(filePath_list)))
    # Get all sample tags
    label_list = []
    for filePath in filePath_list:
        label = filePath.split(os.sep)[-2]
        label_list.append(label)

    labels_set = list(set(label_list))
    print("labels:{}".format(labels_set))

    #标签统计Count
    # print(pd.value_counts(label_list))
    return filePath_list, label_list


def decode_label(label_list,name_table):
    """
    Decode the label according to name_table
    :param label_list:
    :param name_table:
    :return:
    """
    name_list=[]
    for label in label_list:
        name = name_table[label]
        name_list.append(name)
    return name_list


def encode_label(name_list,name_table,unknow=0):
    """
    Code label according to name_table
    :param name_list:
    :param name_table:
    :param unknow : Unknown name, the default label is 0. Generally, in the name_table, index=0 is the background, and the unknown label is also used as the background.
    :return:
    """
    label_list=[]
    for name in name_list:
        if name in name_table:
            index = name_table.index(name)
        else:
            index = unknow
        label_list.append(index)
    return label_list


if __name__=='__main__':
    filename = 'test.txt'
    w_data = [['1.jpg', 'dog', 200, 300, 1.0], ['2.jpg', 'dog', 20, 30, -2]]
    print("w_data=", w_data)
    write_data(filename,w_data, mode='w')
    r_data = read_data(filename)
    print('r_data=', r_data)