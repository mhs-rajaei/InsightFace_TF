import tensorflow as tf
from losses.face_losses import arcface_loss
import tensorlayer as tl
import os
from os.path import join
import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
import datetime
import os
from os.path import join as pjoin
import sys
import copy
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
import facenet
import data.eval_data_reader as eval_data_reader
import verification
import lfw
import time
from sklearn import metrics

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

log_path = os.path.join(PROJECT_PATH, 'output')
models_path = os.path.join(PROJECT_PATH, 'models')


from importlib.machinery import SourceFileLoader
# facenet = SourceFileLoader('facenet', os.path.join(PROJECT_PATH, 'facenet.py')).load_module()
mx2tfrecords = SourceFileLoader('mx2tfrecords', os.path.join(PROJECT_PATH, 'data/mx2tfrecords.py')).load_module()

L_Resnet_E_IR_fix_issue9 = SourceFileLoader('L_Resnet_E_IR_fix_issue9', os.path.join(PROJECT_PATH, 'nets/L_Resnet_E_IR_fix_issue9.py')).load_module()

face_losses = SourceFileLoader('face_losses', os.path.join(PROJECT_PATH, 'losses/face_losses.py')).load_module()
# eval_data_reader = SourceFileLoader('eval_data_reader', os.path.join(PROJECT_PATH, 'data/eval_data_reader.py')).load_module()
# verification = SourceFileLoader('verification', os.path.join(PROJECT_PATH, 'verification.py')).load_module()


class Args:
    net_depth = 50
    epoch = 1000
    lr_steps = [40000, 60000, 80000]
    momentum = 0.9
    weight_decay = 5e-4

    num_output = 85164

    # train_dataset_dir = train_dataset_path
    train_dataset_dir = None
    summary_path = join(log_path, 'summary')
    ckpt_path = join(log_path, 'ckpt')
    log_file_path = join(log_path, 'logs')

    saver_maxkeep = 10
    buffer_size = 10000
    log_device_mapping = False
    summary_interval = 100
    ckpt_interval = 100
    validate_interval = 100
    show_info_interval = 100
    seed = 313
    nrof_preprocess_threads = 4

    ckpt_file = r'F:\Documents\JetBrains\PyCharm\OFR\InsightFace_TF\output\ckpt\model_d\InsightFace_iter_best_'
    ckpt_index_list = ['710000.ckpt']

    # insightface_dataset_dir = eval_dir_path
    insightface_pair = os.path.join(PROJECT_PATH, 'data/First_100_ALL VIS_112_1.txt')
    insightface_dataset_dir = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\First_100_ALL VIS_112"

    insightface_image_size = 112
    batch_size = 32

    facenet_image_size = 160
    facenet_dataset_dir = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\First_100_ALL VIS_160"
    facenet_batch_size = batch_size
    facenet_model = os.path.join(PROJECT_PATH, 'models/facenet/20180402-114759')
    facenet_pairs = insightface_pair

    validation_set_split_ratio = 0.0
    
def data_iter(datasets, batch_size):
    data_num = datasets.shape[0]
    for i in range(0, data_num, batch_size):
        yield datasets[i:min(i+batch_size, data_num), ...]


def get_facenet_embeddings(args):
    # Read the directory containing images
    dataset = facenet.get_dataset(args.facenet_dataset_dir)
    nrof_classes = len(dataset)

    #  Evaluate custom dataset with facenet pre-trained model
    print("Getting embeddings with facenet pre-trained model")
    with tf.Graph().as_default():

        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(dataset)
        assert len(image_list) > 0, 'The  dataset should not be empty'

        print('Number of classes in  dataset: %d' % nrof_classes)
        print('Number of examples in dataset: %d' % len(image_list))

        # Getting batched images by TF dataset
        tf_dataset = facenet.tf_gen_dataset(image_list=image_list, label_list=None, nrof_preprocess_threads=args.nrof_preprocess_threads,
                                            image_size=args.facenet_image_size,  method='cache_slices',
                                            BATCH_SIZE=args.batch_size, repeat_count=1, to_float32=True, shuffle=False)
        tf_dataset_iterator = tf_dataset.make_initializable_iterator()
        tf_dataset_next_element = tf_dataset_iterator.get_next()

        # Getting flip to left_right batched images by TF dataset
        # tf_dataset_flip = facenet.tf_gen_dataset(image_list, label_list, args.nrof_preprocess_threads, args.facenet_image_size,
        #                                     method='cache_slices',
        #                                     BATCH_SIZE=args.batch_size, repeat_count=1, flip_left_right=True, shuffle=False)
        #
        # tf_dataset_iterator_flip = tf_dataset_flip.make_initializable_iterator()
        # tf_dataset_next_element_flip = tf_dataset_iterator_flip.get_next()

        with tf.Session() as sess:
            sess.run(tf_dataset_iterator.initializer)

            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            image_batch = tf.placeholder(name='img_inputs', shape=[None, args.facenet_image_size, args.facenet_image_size, 3], dtype=tf.float32)
            label_batch = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int32)

            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(args.facenet_model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            batch_size = args.batch_size
            input_placeholder = image_batch

            print('getting embeddings..')

            total_time = 0
            batch_number = 0
            embeddings_array = None
            embeddings_array_flip = None
            while True:
                try:
                    images = sess.run(tf_dataset_next_element)

                    data_tmp = images.copy()  # fix issues #4

                    for i in range(data_tmp.shape[0]):
                        data_tmp[i, ...] -= 127.5
                        data_tmp[i, ...] *= 0.0078125
                        data_tmp[i, ...] = cv2.cvtColor(data_tmp[i, ...], cv2.COLOR_RGB2BGR)

                    # Getting flip to left_right batched images by TF dataset
                    data_tmp_flip = images.copy()  # fix issues #4
                    for i in range(data_tmp_flip.shape[0]):
                        data_tmp_flip[i, ...] = np.fliplr(data_tmp_flip[i, ...])
                        data_tmp_flip[i, ...] -= 127.5
                        data_tmp_flip[i, ...] *= 0.0078125
                        data_tmp_flip[i, ...] = cv2.cvtColor(data_tmp_flip[i, ...], cv2.COLOR_RGB2BGR)

                    start_time = time.time()

                    mr_feed_dict = {input_placeholder: data_tmp, phase_train_placeholder: False}
                    mr_feed_dict_flip = {input_placeholder: data_tmp_flip, phase_train_placeholder: False}
                    _embeddings = sess.run(embeddings, mr_feed_dict)
                    _embeddings_flip = sess.run(embeddings, mr_feed_dict_flip)

                    if embeddings_array is None:
                        embeddings_array = np.zeros((len(image_list), _embeddings.shape[1]))
                        embeddings_array_flip = np.zeros((len(image_list), _embeddings_flip.shape[1]))
                    try:
                        embeddings_array[batch_number * batch_size:min((batch_number + 1) * batch_size, len(image_list)), ...] = _embeddings
                        embeddings_array_flip[batch_number * batch_size:min((batch_number + 1) * batch_size, len(image_list)), ...] = _embeddings_flip
                        # print('try: ', batch_number * batch_size, min((batch_number + 1) * batch_size, len(image_list)), ...)
                    except ValueError:
                        print('batch_number*batch_size value is %d min((batch_number+1)*batch_size, len(image_list)) %d,'
                              ' batch_size %d, data.shape[0] %d' %
                              (batch_number * batch_size, min((batch_number + 1) * batch_size, len(image_list)), batch_size, images.shape[0]))
                        print('except: ', batch_number * batch_size, min((batch_number + 1) * batch_size, images.shape[0]), ...)

                    duration = time.time() - start_time
                    batch_number += 1
                    total_time += duration
                except tf.errors.OutOfRangeError:
                    print('tf.errors.OutOfRangeError, Reinitialize tf_dataset_iterator')
                    sess.run(tf_dataset_iterator.initializer)
                    # sess.run(tf_dataset_iterator_flip.initializer)
                    break
    print(f"total_time: {total_time}")

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in [embeddings_array, embeddings_array_flip]:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            # print(_em.shape, _norm)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    final_embeddings_output = embeddings_array + embeddings_array_flip
    final_embeddings_output = sklearn.preprocessing.normalize(final_embeddings_output)
    print(final_embeddings_output.shape)

    return embeddings_array, embeddings_array_flip, final_embeddings_output, _xnorm


def custom_facenet_evaluation(args):
    tf.reset_default_graph()
    # Read the directory containing images
    pairs = read_pairs(args.insightface_pair)
    image_list, issame_list = get_paths_with_pairs(args.facenet_dataset_dir, pairs)

    #  Evaluate custom dataset with facenet pre-trained model
    print("Getting embeddings with facenet pre-trained model")
    with tf.Graph().as_default():
        # Getting batched images by TF dataset
        # image_list = path_list
        tf_dataset = facenet.tf_gen_dataset(image_list=image_list, label_list=None, nrof_preprocess_threads=args.nrof_preprocess_threads,
                                            image_size=args.facenet_image_size,  method='cache_slices',
                                            BATCH_SIZE=args.batch_size, repeat_count=1, to_float32=True, shuffle=False)
        tf_dataset_iterator = tf_dataset.make_initializable_iterator()
        tf_dataset_next_element = tf_dataset_iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf_dataset_iterator.initializer)

            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            image_batch = tf.placeholder(name='img_inputs', shape=[None, args.facenet_image_size, args.facenet_image_size, 3], dtype=tf.float32)
            label_batch = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int32)

            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(args.facenet_model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            batch_size = args.batch_size
            input_placeholder = image_batch

            print('getting embeddings..')

            total_time = 0
            batch_number = 0
            embeddings_array = None
            embeddings_array_flip = None
            while True:
                try:
                    images = sess.run(tf_dataset_next_element)

                    data_tmp = images.copy()  # fix issues #4

                    for i in range(data_tmp.shape[0]):
                        data_tmp[i, ...] -= 127.5
                        data_tmp[i, ...] *= 0.0078125
                        data_tmp[i, ...] = cv2.cvtColor(data_tmp[i, ...], cv2.COLOR_RGB2BGR)

                    # Getting flip to left_right batched images by TF dataset
                    data_tmp_flip = images.copy()  # fix issues #4
                    for i in range(data_tmp_flip.shape[0]):
                        data_tmp_flip[i, ...] = np.fliplr(data_tmp_flip[i, ...])
                        data_tmp_flip[i, ...] -= 127.5
                        data_tmp_flip[i, ...] *= 0.0078125
                        data_tmp_flip[i, ...] = cv2.cvtColor(data_tmp_flip[i, ...], cv2.COLOR_RGB2BGR)

                    start_time = time.time()

                    mr_feed_dict = {input_placeholder: data_tmp, phase_train_placeholder: False}
                    mr_feed_dict_flip = {input_placeholder: data_tmp_flip, phase_train_placeholder: False}
                    _embeddings = sess.run(embeddings, mr_feed_dict)
                    _embeddings_flip = sess.run(embeddings, mr_feed_dict_flip)

                    if embeddings_array is None:
                        embeddings_array = np.zeros((len(image_list), _embeddings.shape[1]))
                        embeddings_array_flip = np.zeros((len(image_list), _embeddings_flip.shape[1]))
                    try:
                        embeddings_array[batch_number * batch_size:min((batch_number + 1) * batch_size, len(image_list)), ...] = _embeddings
                        embeddings_array_flip[batch_number * batch_size:min((batch_number + 1) * batch_size, len(image_list)), ...] = _embeddings_flip
                        # print('try: ', batch_number * batch_size, min((batch_number + 1) * batch_size, len(image_list)), ...)
                    except ValueError:
                        print('batch_number*batch_size value is %d min((batch_number+1)*batch_size, len(image_list)) %d,'
                              ' batch_size %d, data.shape[0] %d' %
                              (batch_number * batch_size, min((batch_number + 1) * batch_size, len(image_list)), batch_size, images.shape[0]))
                        print('except: ', batch_number * batch_size, min((batch_number + 1) * batch_size, images.shape[0]), ...)

                    duration = time.time() - start_time
                    batch_number += 1
                    total_time += duration
                except tf.errors.OutOfRangeError:
                    print('tf.errors.OutOfRangeError, Reinitialize tf_dataset_iterator')
                    sess.run(tf_dataset_iterator.initializer)
                    break

    print(f"total_time: {total_time}")

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in [embeddings_array, embeddings_array_flip]:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            # print(_em.shape, _norm)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    final_embeddings_output = embeddings_array + embeddings_array_flip
    final_embeddings_output = sklearn.preprocessing.normalize(final_embeddings_output)
    print(final_embeddings_output.shape)

    tpr, fpr, accuracy, val, val_std, far = verification.evaluate(final_embeddings_output, issame_list, nrof_folds=10)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)

    auc = metrics.auc(fpr, tpr)
    print('XNorm: %f' % (_xnorm))
    print('Accuracy-Flip: %1.5f+-%1.5f' % (acc2, std2))
    print('TPR: ', np.mean(tpr), 'FPR: ', np.mean(fpr))
    print('Area Under Curve (AUC): %1.3f' % auc)

    tpr_lfw, fpr_lfw, accuracy_lfw, val_lfw, val_std_lfw, far_lfw = lfw.evaluate(final_embeddings_output, issame_list, nrof_folds=10, distance_metric=0,
                                                                                 subtract_mean=False)

    print('accuracy_lfw: %2.5f+-%2.5f' % (np.mean(accuracy_lfw), np.std(accuracy_lfw)))
    print(f"val_lfw: {val_lfw}, val_std_lfw: {val_std_lfw}, far_lfw: {far_lfw}")

    print('val_lfw rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val_lfw, val_std_lfw, far_lfw))
    auc_lfw = metrics.auc(fpr_lfw, tpr_lfw)
    print('TPR_LFW:', np.mean(tpr_lfw), 'FPR_LFW: ', np.mean(fpr_lfw))

    print('Area Under Curve LFW (AUC): %1.3f' % auc_lfw)

    return acc2, std2, _xnorm, [embeddings_array, embeddings_array_flip]


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split('\t')
            if pair not in pairs:
                pairs.append(pair)
    return np.array(pairs)


def get_paths_with_pairs(dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = (os.path.join(dir, pair[0], pair[1]))
            path1 = (os.path.join(dir, pair[0], pair[2]))

            issame = True

        elif len(pair) == 4:
            path0 = os.path.join(dir, pair[0], pair[1])
            path1 = os.path.join(dir, pair[2], pair[3])

            issame = False

        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            print(path0, path1)
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    return path_list, issame_list


def get_insightface_embeddings(args):
    tf.reset_default_graph()
    # Read the directory containing images
    dataset = facenet.get_dataset(args.insightface_dataset_dir)
    nrof_classes = len(dataset)

    #  Evaluate custom dataset with facenet pre-trained model
    print("Getting embeddings with facenet pre-trained model")

    # Get a list of image paths and their labels
    image_list, label_list = facenet.get_image_paths_and_labels(dataset)
    assert len(image_list) > 0, 'The  dataset should not be empty'
    print('Number of classes in  dataset: %d' % nrof_classes)
    print('Number of examples in dataset: %d' % len(image_list))

    # Getting batched images by TF dataset
    tf_dataset = facenet.tf_gen_dataset(image_list=image_list, label_list=None, nrof_preprocess_threads=args.nrof_preprocess_threads,
                                        image_size=args.insightface_dataset_dir, method='cache_slices',
                                        BATCH_SIZE=args.batch_size, repeat_count=1, to_float32=True, shuffle=False)
    tf_dataset_iterator = tf_dataset.make_initializable_iterator()
    tf_dataset_next_element = tf_dataset_iterator.get_next()

    images = tf.placeholder(name='img_inputs', shape=[None, args.insightface_image_size, args.insightface_image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = L_Resnet_E_IR_fix_issue9.get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False,
                                              keep_rate=dropout_rate)
    embeddings = net.outputs
    # mv_mean = tl.layers.get_variables_with_name('resnet_v1_50/bn0/moving_mean', False, True)[0]
    # 3.2 get arcface loss
    logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output)

    sess = tf.Session()
    saver = tf.train.Saver()

    feed_dict = {}
    feed_dict_flip = {}
    path = args.ckpt_file + args.ckpt_index_list[0]
    saver.restore(sess, path)
    print('ckpt file %s restored!' % args.ckpt_index_list[0])
    feed_dict.update(tl.utils.dict_to_one(net.all_drop))
    feed_dict_flip.update(tl.utils.dict_to_one(net.all_drop))
    feed_dict[dropout_rate] = 1.0
    feed_dict_flip[dropout_rate] = 1.0

    batch_size = args.batch_size
    input_placeholder = images

    sess.run(tf_dataset_iterator.initializer)
    # sess.run(tf_dataset_iterator_flip.initializer)
    print('getting embeddings..')

    total_time = 0
    batch_number = 0
    embeddings_array = None
    embeddings_array_flip = None
    while True:
        try:
            images = sess.run(tf_dataset_next_element)

            data_tmp = images.copy()  # fix issues #4

            for i in range(data_tmp.shape[0]):
                data_tmp[i, ...] -= 127.5
                data_tmp[i, ...] *= 0.0078125
                data_tmp[i, ...] = cv2.cvtColor(data_tmp[i, ...], cv2.COLOR_RGB2BGR)

            # Getting flip to left_right batched images by TF dataset
            data_tmp_flip = images.copy()  # fix issues #4
            for i in range(data_tmp_flip.shape[0]):
                data_tmp_flip[i, ...] = np.fliplr(data_tmp_flip[i, ...])
                data_tmp_flip[i, ...] -= 127.5
                data_tmp_flip[i, ...] *= 0.0078125
                data_tmp_flip[i, ...] = cv2.cvtColor(data_tmp_flip[i, ...], cv2.COLOR_RGB2BGR)

            start_time = time.time()

            feed_dict[input_placeholder] = data_tmp
            _embeddings = sess.run(embeddings, feed_dict)

            feed_dict_flip[input_placeholder] = data_tmp_flip
            _embeddings_flip = sess.run(embeddings, feed_dict_flip)

            if embeddings_array is None:
                embeddings_array = np.zeros((len(image_list), _embeddings.shape[1]))
                embeddings_array_flip = np.zeros((len(image_list), _embeddings_flip.shape[1]))
            try:
                embeddings_array[batch_number * batch_size:min((batch_number + 1) * batch_size, len(image_list)), ...] = _embeddings
                embeddings_array_flip[batch_number * batch_size:min((batch_number + 1) * batch_size, len(image_list)),
                ...] = _embeddings_flip
                # print('try: ', batch_number * batch_size, min((batch_number + 1) * batch_size, len(image_list)), ...)
            except ValueError:
                print('batch_number*batch_size value is %d min((batch_number+1)*batch_size, len(image_list)) %d,'
                      ' batch_size %d, data.shape[0] %d' %
                      (batch_number * batch_size, min((batch_number + 1) * batch_size, len(image_list)), batch_size,
                       images.shape[0]))
                print('except: ', batch_number * batch_size, min((batch_number + 1) * batch_size, images.shape[0]), ...)

            duration = time.time() - start_time
            batch_number += 1
            total_time += duration
        except tf.errors.OutOfRangeError:
            print('tf.errors.OutOfRangeError, Reinitialize tf_dataset_iterator')
            sess.run(tf_dataset_iterator.initializer)
            break

    print(f"total_time: {total_time}")

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in [embeddings_array, embeddings_array_flip]:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            # print(_em.shape, _norm)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    final_embeddings_output = embeddings_array + embeddings_array_flip
    final_embeddings_output = sklearn.preprocessing.normalize(final_embeddings_output)
    print(final_embeddings_output.shape)

    sess.close()

    return embeddings_array, embeddings_array_flip, final_embeddings_output, _xnorm


def custom_insightface_evaluation(args):
    tf.reset_default_graph()
    # Read the directory containing images
    pairs = read_pairs(args.insightface_pair)
    image_list, issame_list = get_paths_with_pairs(args.insightface_dataset_dir, pairs)

    #  Evaluate custom dataset with facenet pre-trained model
    print("Getting embeddings with facenet pre-trained model")

    # Getting batched images by TF dataset
    tf_dataset = facenet.tf_gen_dataset(image_list=image_list, label_list=None, nrof_preprocess_threads=args.nrof_preprocess_threads,
                                        image_size=args.insightface_dataset_dir, method='cache_slices',
                                        BATCH_SIZE=args.batch_size, repeat_count=1, to_float32=True, shuffle=False)
    # tf_dataset = facenet.tf_gen_dataset(image_list, label_list, args.nrof_preprocess_threads, args.facenet_image_size, method='cache_slices',
    #                                     BATCH_SIZE=args.batch_size, repeat_count=1, shuffle=False)
    tf_dataset_iterator = tf_dataset.make_initializable_iterator()
    tf_dataset_next_element = tf_dataset_iterator.get_next()

    images = tf.placeholder(name='img_inputs', shape=[None, args.insightface_image_size, args.insightface_image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = L_Resnet_E_IR_fix_issue9.get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False,
                                              keep_rate=dropout_rate)
    embeddings = net.outputs
    # mv_mean = tl.layers.get_variables_with_name('resnet_v1_50/bn0/moving_mean', False, True)[0]
    # 3.2 get arcface loss
    logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output)

    sess = tf.Session()
    saver = tf.train.Saver()

    feed_dict = {}
    feed_dict_flip = {}
    path = args.ckpt_file + args.ckpt_index_list[0]
    saver.restore(sess, path)
    print('ckpt file %s restored!' % args.ckpt_index_list[0])
    feed_dict.update(tl.utils.dict_to_one(net.all_drop))
    feed_dict_flip.update(tl.utils.dict_to_one(net.all_drop))
    feed_dict[dropout_rate] = 1.0
    feed_dict_flip[dropout_rate] = 1.0

    batch_size = args.batch_size
    input_placeholder = images

    sess.run(tf_dataset_iterator.initializer)
    print('getting embeddings..')

    total_time = 0
    batch_number = 0
    embeddings_array = None
    embeddings_array_flip = None
    while True:
        try:
            images = sess.run(tf_dataset_next_element)

            data_tmp = images.copy()  # fix issues #4

            for i in range(data_tmp.shape[0]):
                data_tmp[i, ...] -= 127.5
                data_tmp[i, ...] *= 0.0078125
                data_tmp[i, ...] = cv2.cvtColor(data_tmp[i, ...], cv2.COLOR_RGB2BGR)

            # Getting flip to left_right batched images by TF dataset
            data_tmp_flip = images.copy()  # fix issues #4
            for i in range(data_tmp_flip.shape[0]):
                data_tmp_flip[i, ...] = np.fliplr(data_tmp_flip[i, ...])
                data_tmp_flip[i, ...] -= 127.5
                data_tmp_flip[i, ...] *= 0.0078125
                data_tmp_flip[i, ...] = cv2.cvtColor(data_tmp_flip[i, ...], cv2.COLOR_RGB2BGR)

            start_time = time.time()

            feed_dict[input_placeholder] = data_tmp
            _embeddings = sess.run(embeddings, feed_dict)

            feed_dict_flip[input_placeholder] = data_tmp_flip
            _embeddings_flip = sess.run(embeddings, feed_dict_flip)

            if embeddings_array is None:
                embeddings_array = np.zeros((len(image_list), _embeddings.shape[1]))
                embeddings_array_flip = np.zeros((len(image_list), _embeddings_flip.shape[1]))
            try:
                embeddings_array[batch_number * batch_size:min((batch_number + 1) * batch_size, len(image_list)), ...] = _embeddings
                embeddings_array_flip[batch_number * batch_size:min((batch_number + 1) * batch_size, len(image_list)),
                ...] = _embeddings_flip
                # print('try: ', batch_number * batch_size, min((batch_number + 1) * batch_size, len(image_list)), ...)
            except ValueError:
                print('batch_number*batch_size value is %d min((batch_number+1)*batch_size, len(image_list)) %d,'
                      ' batch_size %d, data.shape[0] %d' %
                      (batch_number * batch_size, min((batch_number + 1) * batch_size, len(image_list)), batch_size,
                       images.shape[0]))
                print('except: ', batch_number * batch_size, min((batch_number + 1) * batch_size, images.shape[0]), ...)

            duration = time.time() - start_time
            batch_number += 1
            total_time += duration
        except tf.errors.OutOfRangeError:
            print('tf.errors.OutOfRangeError, Reinitialize tf_dataset_iterator')
            sess.run(tf_dataset_iterator.initializer)
            break
    
    print(f"total_time: {total_time}")

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in [embeddings_array, embeddings_array_flip]:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            # print(_em.shape, _norm)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    final_embeddings_output = embeddings_array + embeddings_array_flip
    final_embeddings_output = sklearn.preprocessing.normalize(final_embeddings_output)
    print(final_embeddings_output.shape)

    tpr, fpr, accuracy, val, val_std, far = verification.evaluate(final_embeddings_output, issame_list, nrof_folds=10)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)

    auc = metrics.auc(fpr, tpr)
    print('XNorm: %f' % (_xnorm))
    print('Accuracy-Flip: %1.5f+-%1.5f' % (acc2, std2))
    print('TPR: ', np.mean(tpr), 'FPR: ', np.mean(fpr))
    print('Area Under Curve (AUC): %1.3f' % auc)

    tpr_lfw, fpr_lfw, accuracy_lfw, val_lfw, val_std_lfw, far_lfw = lfw.evaluate(final_embeddings_output, issame_list, nrof_folds=10,
                                                                                 distance_metric=0,
                                                                                 subtract_mean=False)

    print('accuracy_lfw: %2.5f+-%2.5f' % (np.mean(accuracy_lfw), np.std(accuracy_lfw)))
    print(f"val_lfw: {val_lfw}, val_std_lfw: {val_std_lfw}, far_lfw: {far_lfw}")

    print('val_lfw rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val_lfw, val_std_lfw, far_lfw))
    auc_lfw = metrics.auc(fpr_lfw, tpr_lfw)
    print('TPR_LFW:', np.mean(tpr_lfw), 'FPR_LFW: ', np.mean(fpr_lfw))

    print('Area Under Curve LFW (AUC): %1.3f' % auc_lfw)

    sess.close()

    return acc2, std2, _xnorm, [embeddings_array, embeddings_array_flip]


if __name__ == '__main__':
    args = Args()
    get_facenet_embeddings(args)
    print(f"{'*@*'}"*50)
    print(f"{'*@*'}"*50)
    print(f"{'*@*'}"*50)
    get_insightface_embeddings(args)
