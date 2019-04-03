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
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import classifier


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
    # facenet_dataset_dir = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\First_70_ALL NIR_160"
    facenet_dataset_dir = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\All VIS+NIR_160"
    facenet_batch_size = batch_size
    facenet_model = os.path.join(PROJECT_PATH, 'models/facenet/20180402-114759')
    facenet_pairs = insightface_pair

    validation_set_split_ratio = 0.75
    min_nrof_val_images_per_class = 5
    classifier = "knn"  # svm or knn
    use_trained_svm = None


def data_iter(datasets, batch_size):
    data_num = datasets.shape[0]
    for i in range(0, data_num, batch_size):
        yield datasets[i:min(i+batch_size, data_num), ...]


def get_facenet_embeddings(args, image_list=None, label_list=None):
    if image_list is None:
        # Read the directory containing images
        dataset = facenet.get_dataset(args.facenet_dataset_dir)
        nrof_classes = len(dataset)

    #  Evaluate custom dataset with facenet pre-trained model
    print("Getting embeddings with facenet pre-trained model")
    with tf.Graph().as_default():
        if image_list is None:
            # Get a list of image paths and their labels
            image_list, label_list = facenet.get_image_paths_and_labels(dataset)
            print('Number of classes in  dataset: %d' % nrof_classes)

        assert len(image_list) > 0, 'The  dataset should not be empty'

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


def get_insightface_embeddings(args, image_list=None, label_list=None):
    tf.reset_default_graph()
    if image_list is None:
        # Read the directory containing images
        dataset = facenet.get_dataset(args.insightface_dataset_dir)
        nrof_classes = len(dataset)

    #  Evaluate custom dataset with facenet pre-trained model
    print("Getting embeddings with facenet pre-trained model")

    if image_list is None:
        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(dataset)
        print('Number of classes in  dataset: %d' % nrof_classes)

    assert len(image_list) > 0, 'The  dataset should not be empty'
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
    # logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output)

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

    sess.close()

    return embeddings_array, embeddings_array_flip, final_embeddings_output, _xnorm


def test_model(embeddings_array, embeddings_array_flip, final_embeddings_output, dataset=None, image_list=None, label_list=None,
               name_dict=None, index_dict=None, facenet_insightface='facenet', nrof_classes=None):
    # if dataset:
    # Get a list of image paths and their labels
    _image_list, _label_list, _name_dict, _index_dict = facenet.get_image_paths_and_labels(dataset, path=True)

    # Get embedding of _image_list
    _embeddings_array, _embeddings_array_flip, _final_embeddings_output, xnorm = get_facenet_embeddings(args, image_list=_image_list)

    # @#$#$%$%^&^&*&*(&*(Q@@!#@#$!@#$#$%@$#^%$&^%&^&*^&*()!@#!@#$#$%$%^%^&%^*^&(&*)*()!@#$@$#$%$%^$%&^&*^(&)*!@#!@#$$#@$#$%$%^%^&%^*^&*(**&^*(
    # Run Classification
    if args.use_trained_svm == None:
        args.use_trained_svm = ""

    start_time_classify = time.time()
    result = classify(args.classifier, args.use_trained_svm, final_embeddings_output, label_list, _final_embeddings_output, _label_list,
                      nrof_classes_facenet, index_dict)

    print("Classify Time: %s minutes" % ((time.time() - start_time_classify) / 60))


def classify(classify_type, trained_svm, train_data, train_labels, test_data, test_labels, num_classes, label_lookup_dict):
    """
    classify - function to use facial embeddings to judge what label a face is associated with

    args    classify_type - type of classification to use ("svm" or "knn")
            train_data - data to use for training
            train_labels - labels to use for training
            test_data - data to use for testing
            test_labels - labels to check against predicted values
            num_classes - required for neural classifier
            label_lookup_dict - dict for easy lookup of int to label

    returns accuracy - accuracy of the produced model
    """

    if classify_type == "svm":
        classify_method = classifier.SVM_Classifier(train_data, train_labels, test_data, test_labels)
    elif classify_type == "neural":
        classify_method = classifier.Neural_Classifier(train_data, train_labels, test_data, test_labels, num_classes)
    elif classify_type == "knn":
        classify_method = classifier.KNNClassifier(train_data, train_labels, test_data, test_labels)
    else:
        print("You have provided and invalid classifier type. (Valid options are svm or neural)")
        return False

    #if we are provided with a pre trained svm, there is no need to carry out training
    if trained_svm == "":
        model = classify_method.train()
    else:
        print("Using pre trained svm...")
        model = joblib.load(trained_svm)

    accuracy = classify_method.check_accuracy(model, label_lookup_dict)

    return accuracy


if __name__ == '__main__':

    args = Args()

    if args.validation_set_split_ratio > 0.0:
        # Read the directory containing images
        insightface_dataset = facenet.get_dataset(args.insightface_dataset_dir)
        nrof_classes_insightface = len(insightface_dataset)

        # Read the directory containing images
        facenet_dataset = facenet.get_dataset(args.facenet_dataset_dir)
        nrof_classes_facenet = len(facenet_dataset)

        # Split dataset to train and validation set's
        train_set_insightface, val_set_insightface = facenet.split_dataset(insightface_dataset, args.validation_set_split_ratio,
                                                                           args.min_nrof_val_images_per_class, 'SPLIT_IMAGES')
        train_set_facenet, val_set_facenet = facenet.split_dataset(facenet_dataset, args.validation_set_split_ratio, args.min_nrof_val_images_per_class,
                                                                   'SPLIT_IMAGES')
        # _val_set_facenet = facenet.get_dataset(r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\First_70_ALL VIS_160")
        # print(f"len(val_set_facenet): {len(val_set_facenet)}")
        # # InsightFace
        # # Get a list of image paths and their labels
        # image_list_insightface, label_list_insightface, name_dict_insightface, index_dict_insightface = \
        #     facenet.get_image_paths_and_labels(train_set_insightface,path=args.insightface_dataset_dir)
        # # Get embedding of database
        # embeddings_array_insightface, embeddings_array_flip_insightface, final_embeddings_output_insightface, xnorm_insightface = \
        #     get_facenet_embeddings(args, image_list=image_list_insightface)

        # FaceNet
        # Get a list of image paths and their labels
        image_list_facenet, label_list_facenet, name_dict_facenet, index_dict_facenet = \
            facenet.get_image_paths_and_labels(train_set_facenet, path=args.facenet_dataset_dir)
        # Get embedding of database
        embeddings_array_facenet, embeddings_array_flip_facenet, final_embeddings_output_facenet, xnorm_facenet = \
            get_facenet_embeddings(args, image_list=image_list_facenet)

        # @#@##@##@#@@#@#@#@#@#@@#@@#@##@#@#@#@#@#@#@#@#@#@#@#@#@@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@@#@
        # Test the model
        test_model(embeddings_array_facenet, embeddings_array_flip_facenet, final_embeddings_output_facenet, dataset=val_set_facenet,
                   image_list=image_list_facenet, label_list=label_list_facenet,
                   name_dict=name_dict_facenet, index_dict=index_dict_facenet, nrof_classes=nrof_classes_facenet)


    else:
        get_facenet_embeddings(args)
        print(f"{'*@*'}" * 50)
        print(f"{'*@*'}" * 50)
        print(f"{'*@*'}" * 50)
        get_insightface_embeddings(args)


