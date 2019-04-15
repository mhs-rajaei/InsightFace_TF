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
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from face_recognition_knn import *


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


def data_iter(datasets, batch_size):
    data_num = datasets.shape[0]
    for i in range(0, data_num, batch_size):
        yield datasets[i:min(i+batch_size, data_num), ...]


class FaceNet:
    def __init__(self, args, graph=None, embeddings_array=None, embeddings_array_flip=None, final_embeddings_output=None, xnorm=None, sess=None,
                 image_batch=None, label_batch=None, phase_train_placeholder=None, input_map=None, embeddings=None,
                 image_list=None, label_list=None):
        self.args = args
        self.graph = graph
        self.embeddings_array = embeddings_array
        self.embeddings_array_flip = embeddings_array_flip
        self.final_embeddings_output = final_embeddings_output
        self.xnorm = xnorm
        self.sess = sess
        self.image_batch = image_batch
        self.label_batch = label_batch
        self.phase_train_placeholder = phase_train_placeholder
        self.input_map = input_map
        self.embeddings = embeddings
        self.label_list = label_list
        self.pre_trained_model_loaded = False
        self.image_list = image_list

    def get_embeddings(self):
        #  Evaluate custom dataset with FaceNet pre-trained model
        print("Getting embeddings with FaceNet pre-trained model")
        # with tf.Graph().as_default():
        if self.graph is None:
            self.graph = tf.Graph()
        with self.graph.as_default():
            if self.image_list is None:
                # Read the directory containing images
                dataset = facenet.get_dataset(self.args.facenet_dataset_dir)
                nrof_classes = len(dataset)
                # Get a list of image paths and their labels
                self.image_list, self.label_list = facenet.get_image_paths_and_labels(dataset)
                print('Number of classes in  dataset: %d' % nrof_classes)

            assert len(self.image_list) > 0, 'The  dataset should not be empty'

            print('Number of examples in dataset: %d' % len(self.image_list))

            # Getting batched images by TF dataset
            tf_dataset = facenet.tf_gen_dataset(image_list=self.image_list, label_list=None,
                                                nrof_preprocess_threads=self.args.nrof_preprocess_threads,
                                                image_size=self.args.facenet_image_size,  method='cache_slices',
                                                BATCH_SIZE=self.args.batch_size, repeat_count=1, to_float32=True, shuffle=False)
            tf_dataset_iterator = tf_dataset.make_initializable_iterator()
            tf_dataset_next_element = tf_dataset_iterator.get_next()

            if self.sess is None:
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=self.args.log_device_placement))

            self.sess.run(tf_dataset_iterator.initializer)

            if self.phase_train_placeholder is None:
                self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            if self.image_batch is None:
                self.image_batch = tf.placeholder(name='img_inputs', shape=[None, self.args.facenet_image_size, self.args.facenet_image_size, 3],
                                                  dtype=tf.float32)
            if self.label_batch is None:
                self.label_batch = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int32)

            # Load the model
            if self.input_map is None:
                self.input_map = {'image_batch': self.image_batch, 'label_batch': self.label_batch, 'phase_train': self.phase_train_placeholder}

            if not self.pre_trained_model_loaded:
                facenet.load_model(self.args.facenet_model, input_map=self.input_map, session=self.sess)
                self.pre_trained_model_loaded = True

            # Get output tensor
            if self.embeddings is None:
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            batch_size = self.args.batch_size

            print('getting embeddings..')

            total_time = 0
            batch_number = 0
            embeddings_array = None
            embeddings_array_flip = None
            while True:
                try:
                    images = self.sess.run(tf_dataset_next_element)

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

                    mr_feed_dict = {self.image_batch: data_tmp, self.phase_train_placeholder: False}
                    mr_feed_dict_flip = {self.image_batch: data_tmp_flip, self.phase_train_placeholder: False}
                    _embeddings = self.sess.run(self.embeddings, mr_feed_dict)
                    _embeddings_flip = self.sess.run(self.embeddings, mr_feed_dict_flip)

                    if embeddings_array is None:
                        embeddings_array = np.zeros((len(self.image_list), _embeddings.shape[1]))
                        embeddings_array_flip = np.zeros((len(self.image_list), _embeddings_flip.shape[1]))
                    try:
                        embeddings_array[batch_number * batch_size:min((batch_number + 1) * batch_size, len(self.image_list)), ...] = _embeddings
                        embeddings_array_flip[batch_number * batch_size:min((batch_number + 1) * batch_size, len(self.image_list)),
                        ...] = _embeddings_flip
                        # print('try: ', batch_number * batch_size, min((batch_number + 1) * batch_size, len(image_list)), ...)
                    except ValueError:
                        print('batch_number*batch_size value is %d min((batch_number+1)*batch_size, len(image_list)) %d,'
                              ' batch_size %d, data.shape[0] %d' %
                              (batch_number * batch_size, min((batch_number + 1) * batch_size, len(self.image_list)), batch_size, images.shape[0]))
                        print('except: ', batch_number * batch_size, min((batch_number + 1) * batch_size, images.shape[0]), ...)

                    duration = time.time() - start_time
                    batch_number += 1
                    total_time += duration
                except tf.errors.OutOfRangeError:
                    print('tf.errors.OutOfRangeError, Reinitialize tf_dataset_iterator')
                    self.sess.run(tf_dataset_iterator.initializer)
                    break
        print(f"total_time: {total_time}")

        xnorm = 0.0
        xnorm_cnt = 0
        for embed in [embeddings_array, embeddings_array_flip]:
            for i in range(embed.shape[0]):
                _em = embed[i]
                _norm = np.linalg.norm(_em)
                # print(_em.shape, _norm)
                xnorm += _norm
                xnorm_cnt += 1
        xnorm /= xnorm_cnt

        final_embeddings_output = embeddings_array + embeddings_array_flip
        final_embeddings_output = sklearn.preprocessing.normalize(final_embeddings_output)
        print(final_embeddings_output.shape)

        return embeddings_array, embeddings_array_flip, final_embeddings_output, xnorm


class InsightFace:
    def __init__(self, args, graph=None, embeddings_array=None, embeddings_array_flip=None, final_embeddings_output=None, xnorm=None, sess=None,
                 image_batch=None, label_batch=None, embeddings=None, image_list=None, label_list=None, dropout_rate=None,
                 w_init_method=None, net=None, saver=None, feed_dict=None,
                 feed_dict_flip=None):
        self.args = args
        self.graph = graph
        self.embeddings_array = embeddings_array
        self.embeddings_array_flip = embeddings_array_flip
        self.final_embeddings_output = final_embeddings_output
        self.xnorm = xnorm
        self.sess = sess
        self.image_batch = image_batch
        self.label_batch = label_batch
        self.embeddings = embeddings
        self.pre_trained_model_loaded = False
        self.image_list = image_list
        self.label_list = label_list
        self.dropout_rate = dropout_rate
        self.w_init_method = w_init_method
        self.net = net
        self.embeddings = embeddings
        self.saver = saver
        self.feed_dict = feed_dict
        self.feed_dict_flip = feed_dict_flip

    def get_embeddings(self):
        #  Evaluate custom dataset with InsightFace pre-trained model
        print("Getting embeddings with InsightFace pre-trained model")

        if self.graph is None:
            self.graph = tf.Graph()

        with self.graph.as_default():
            if self.image_list is None:
                # Read the directory containing images
                dataset = facenet.get_dataset(self.args.facenet_dataset_dir)
                nrof_classes = len(dataset)
                # Get a list of image paths and their labels
                self.image_list, self.label_list = facenet.get_image_paths_and_labels(dataset)
                print('Number of classes in  dataset: %d' % nrof_classes)

            assert len(self.image_list) > 0, 'The  dataset should not be empty'
            print('Number of examples in dataset: %d' % len(self.image_list))

            # Getting batched images by TF dataset
            tf_dataset = facenet.tf_gen_dataset(image_list=self.image_list, label_list=None, nrof_preprocess_threads=self.args.nrof_preprocess_threads,
                                                image_size=self.args.insightface_dataset_dir, method='cache_slices',
                                                BATCH_SIZE=self.args.batch_size, repeat_count=1, to_float32=True, shuffle=False)
            tf_dataset_iterator = tf_dataset.make_initializable_iterator()
            tf_dataset_next_element = tf_dataset_iterator.get_next()

            if self.image_batch is None:
                self.image_batch = tf.placeholder(name='img_inputs', shape=[None, self.args.insightface_image_size, self.args.insightface_image_size, 3],
                                    dtype=tf.float32)
            if self.label_batch is None:
                self.label_batch = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
            if self.dropout_rate is None:
                self.dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
            if self.w_init_method is None:
                self.w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
            if self.net is None:
                self.net = L_Resnet_E_IR_fix_issue9.get_resnet(self.image_batch, self.args.net_depth, type='ir', w_init=self.w_init_method,
                                                               trainable=False, keep_rate=self.dropout_rate)
            if self.embeddings is None:
                self.embeddings = self.net.outputs
            # mv_mean = tl.layers.get_variables_with_name('resnet_v1_50/bn0/moving_mean', False, True)[0]
            # 3.2 get arcface loss
            # logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=self.args.num_output)
            if self.sess is None:
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=self.args.log_device_placement))
            if self.saver is None:
                self.saver = tf.train.Saver()

            self.feed_dict = {}
            self.feed_dict_flip = {}

            if not self.pre_trained_model_loaded:
                path = self.args.ckpt_file + self.args.ckpt_index_list[0]
                self.saver.restore(self.sess, path)
                self.pre_trained_model_loaded = True
                print('ckpt file %s restored!' % self.args.ckpt_index_list[0])

            self.feed_dict.update(tl.utils.dict_to_one(self.net.all_drop))
            self.feed_dict_flip.update(tl.utils.dict_to_one(self.net.all_drop))
            self.feed_dict[self.dropout_rate] = 1.0
            self.feed_dict_flip[self.dropout_rate] = 1.0

            batch_size = self.args.batch_size

            self.sess.run(tf_dataset_iterator.initializer)
            print('getting embeddings..')

            total_time = 0
            batch_number = 0
            embeddings_array = None
            embeddings_array_flip = None
            while True:
                try:
                    images = self.sess.run(tf_dataset_next_element)

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

                    self.feed_dict[self.image_batch] = data_tmp
                    _embeddings = self.sess.run(self.embeddings, self.feed_dict)

                    self.feed_dict_flip[self.image_batch] = data_tmp_flip
                    _embeddings_flip = self.sess.run(self.embeddings, self.feed_dict_flip)

                    if embeddings_array is None:
                        embeddings_array = np.zeros((len(self.image_list), _embeddings.shape[1]))
                        embeddings_array_flip = np.zeros((len(self.image_list), _embeddings_flip.shape[1]))
                    try:
                        embeddings_array[batch_number * batch_size:min((batch_number + 1) * batch_size, len(self.image_list)), ...] = _embeddings
                        embeddings_array_flip[batch_number * batch_size:min((batch_number + 1) * batch_size, len(self.image_list)),
                        ...] = _embeddings_flip
                        # print('try: ', batch_number * batch_size, min((batch_number + 1) * batch_size, len(image_list)), ...)
                    except ValueError:
                        print('batch_number*batch_size value is %d min((batch_number+1)*batch_size, len(image_list)) %d,'
                              ' batch_size %d, data.shape[0] %d' %
                              (batch_number * batch_size, min((batch_number + 1) * batch_size, len(self.image_list)), batch_size,
                               images.shape[0]))
                        print('except: ', batch_number * batch_size, min((batch_number + 1) * batch_size, images.shape[0]), ...)

                    duration = time.time() - start_time
                    batch_number += 1
                    total_time += duration
                except tf.errors.OutOfRangeError:
                    print('tf.errors.OutOfRangeError, Reinitialize tf_dataset_iterator')
                    self.sess.run(tf_dataset_iterator.initializer)
                    break

            print(f"total_time: {total_time}")

        xnorm = 0.0
        xnorm_cnt = 0
        for embed in [embeddings_array, embeddings_array_flip]:
            for i in range(embed.shape[0]):
                _em = embed[i]
                _norm = np.linalg.norm(_em)
                # print(_em.shape, _norm)
                xnorm += _norm
                xnorm_cnt += 1
        xnorm /= xnorm_cnt

        final_embeddings_output = embeddings_array + embeddings_array_flip
        final_embeddings_output = sklearn.preprocessing.normalize(final_embeddings_output)
        print(final_embeddings_output.shape)

        return embeddings_array, embeddings_array_flip, final_embeddings_output, xnorm


def test_model(args, facenet_or_insightface='facenet'):

    if facenet_or_insightface == 'facenet':
        class_obj = FaceNet(args)
        dataset_dir = args.facenet_dataset_dir
        val_dataset_dir = args.facenet_val_dataset_dir
    else:
        class_obj = InsightFace(args)
        dataset_dir = args.insightface_dataset_dir
        val_dataset_dir = args.insightface_val_dataset_dir

    # Read the directory containing images
    dataset = facenet.get_dataset(dataset_dir)
    nrof_classes = len(dataset)

    if args.validation_set_split_ratio > 0.0:
        # Split dataset to train and validation set's
        train_set, val_set = facenet.split_dataset(dataset, args.validation_set_split_ratio, args.min_nrof_val_images_per_class, 'SPLIT_IMAGES')
    else:
        train_set = dataset
        val_set = facenet.get_dataset(val_dataset_dir)
        nrof_val_classes = len(val_set)

    # Get a list of image paths and their labels
    image_list, label_list, name_dict, index_dict = facenet.get_image_paths_and_labels(train_set, path=True)

    class_obj.image_list = image_list

    # Get embedding of _image_list
    embeddings_array, embeddings_array_flip, final_embeddings_output, xnorm = class_obj.get_embeddings()

    # @#@##@##@#@@#@#@#@#@#@@#@@#@##@#@#@#@#@#@#@#@#@#@#@#@#@@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@@#@
    # Test the model
    # Get a list of image paths and their labels
    test_image_list, test_label_list, test_name_dict, test_index_dict = facenet.get_image_paths_and_labels(val_set, path=True)
    class_obj.image_list = test_image_list
    # Get embedding of _image_list
    test_embeddings_array, test_embeddings_array_flip, test_final_embeddings_output, test_xnorm = class_obj.get_embeddings()

    # Run Classification
    if args.use_trained_svm == None:
        args.use_trained_svm = ""

    # What is the best threshold for the verification problem (Distance Threshold)
    from sklearn.metrics import f1_score, accuracy_score
    distances = []  # squared L2 distance between pairs
    identical = []  # 1 if same identity, 0 otherwise

    for i in range(len(label_list)):
        for j in range(len(test_label_list)):
            distances.append(distance(final_embeddings_output[i], test_final_embeddings_output[j]))
            identical.append(1 if label_list[i] == test_label_list[j] else 0)

    distances = np.array(distances)
    identical = np.array(identical)

    thresholds = np.arange(0.3, 1.0, 0.01)

    f1_scores = [f1_score(identical, distances < t) for t in thresholds]
    acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

    opt_idx = np.argmax(f1_scores)
    # Threshold at maximal F1 score
    opt_tau = thresholds[opt_idx]
    # Accuracy at maximal F1 score
    opt_acc = accuracy_score(identical, distances < opt_tau)

    # Plot F1 score and accuracy as function of distance threshold
    plt.plot(thresholds, f1_scores, label='F1 score')
    plt.plot(thresholds, acc_scores, label='Accuracy')
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}')
    plt.xlabel('Distance threshold')
    plt.legend()
    plt.show()
    # -------------------------------------------------------------------------------------------------------------------------


    # Distance distributions of positive and negative pairs
    dist_pos = distances[identical == 1]
    dist_neg = distances[identical == 0]

    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.hist(dist_pos)
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title('Distances (pos. pairs)')
    plt.legend()

    plt.subplot(122)
    plt.hist(dist_neg)
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title('Distances (neg. pairs)')
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------------------------------------------------------

    # Face recognition - with KNN or an SVM
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    svc = LinearSVC()

    knn.fit(final_embeddings_output, label_list)
    svc.fit(final_embeddings_output, label_list)

    y_pred_knn = knn.predict(test_final_embeddings_output)
    acc_knn = accuracy_score(test_label_list, y_pred_knn)
    y_pred_svc = knn.predict(test_final_embeddings_output)
    acc_svc = accuracy_score(test_label_list, y_pred_svc)

    print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')

    # -------------------------------------------------------------------------------------------------------------------------
    K = 1
    knn_2 = NearestNeighbors(n_neighbors=K)
    knn_2.fit(final_embeddings_output, label_list)
    dist, ind = knn_2.kneighbors()
    density_knn = 1 / (np.sum(dist, axis=1) / K)

    pca = PCA()
    pca.fit(final_embeddings_output, label_list)
    pcas = np.inner(final_embeddings_output, pca.components_)

    knn_treshold = opt_tau
    plt.scatter(pcas[:, 0], pcas[:, 1], c=np.where(density_knn < knn_treshold, 0, 1))
    plt.show()

    def visualize_density(density, filename=None, count=None):
        if count is None:
            count = 5
        indices = np.argsort(density)[:count]
        x = np.arange(len(indices))
        plt.ylabel('Density')
        plt.yscale('log')
        plt.xlabel('Outlier number')
        plt.bar(x, density[indices], bottom=np.min(density) / 100)
        if filename:
            plt.savefig(filename)
        plt.show()

    visualize_density(density_knn, filename=None, count=len(test_label_list))

    # -------------------------------------------------------------------------------------------------------------------------

    classifier = train(X=final_embeddings_output, y=label_list, n_neighbors=1)
    predictions = predict(test_final_embeddings_output, knn_clf=classifier, distance_threshold=0.8)
    acc_3 = 0
    for i in range(len(predictions)):
        if predictions[i][0] == test_label_list[i]:
            acc_3 += 1
    acc_3 /= len(predictions)
    print()
    # -------------------------------------------------------------------------------------------------------------------------

    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    def load_image(path):
        img = cv2.imread(path, 1)
        return img[..., ::-1]

    import warnings
    # Suppress LabelEncoder warning
    warnings.filterwarnings('ignore')

    def show_prediction(example_idx, label):
        plt.figure()
        example_image = load_image(test_image_list[example_idx])
        example_prediction = knn.predict([test_final_embeddings_output[example_idx]])
        encoder = LabelEncoder()
        encoder.fit(test_label_list)
        example_identity = encoder.inverse_transform(example_prediction)[0]

        plt.imshow(example_image)
        plt.title(f'Recognized as {example_identity}, Correct label is {label}')
        plt.show()

    def show_predictions(indexes):
        plt.figure(figsize=(16, 16))

        for i, idx in enumerate(indexes[:16]):
            example_image = load_image(test_image_list[idx])
            example_prediction = knn.predict([test_final_embeddings_output[idx]])
            encoder = LabelEncoder()
            encoder.fit(test_label_list)
            example_identity = encoder.inverse_transform(example_prediction)[0]

            plt.subplot(4, 4, i + 1)
            plt.imshow(example_image)
            plt.title(f'Recognized as {example_identity+1}, Correct label is {test_label_list[idx]+1}')
        plt.show()

    idxs = range(0, len(test_image_list))
    random_idxs = random.sample(idxs, 16)
    show_predictions(random_idxs)
    # -------------------------------------------------------------------------------------------------------------------------
    # Missclassified images
    error_pairs = []
    for i, item in enumerate(y_pred_knn):
        if item != test_label_list[i]:
            error_pairs.append(i)

    print(error_pairs)

    random_error_pairs_idxs = random.sample(error_pairs, 16)
    show_predictions(random_error_pairs_idxs)
    # -------------------------------------------------------------------------------------------------------------------------

    # Dataset visualization

    # -------------------------------------------------------------------------------------------------------------------------

    start_time_classify = time.time()
    result = classify(args.classifier, args.use_trained_svm, final_embeddings_output, label_list, test_final_embeddings_output, test_label_list,
                      nrof_classes, index_dict)

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


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


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
    insightface_val_dataset_dir = None

    insightface_image_size = 112
    batch_size = 32

    facenet_image_size = 160
    facenet_dataset_dir = r'E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\First_70_ALL VIS_160'
    facenet_val_dataset_dir = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\First_70_ALL NIR_160"
    # facenet_dataset_dir = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\All VIS+NIR_160"
    facenet_batch_size = batch_size
    facenet_model = os.path.join(PROJECT_PATH, 'models/facenet/20180402-114759')
    facenet_pairs = insightface_pair

    validation_set_split_ratio = 0.0
    min_nrof_val_images_per_class = 1
    classifier = "knn"  # svm or knn
    use_trained_svm = None

    log_device_placement = False


if __name__ == '__main__':

    args = Args()
    test_model(args, facenet_or_insightface='facenet')
    # test_model(args, facenet_or_insightface='insightfface')

