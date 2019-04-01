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

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

log_path = os.path.join(PROJECT_PATH, 'output')
models_path = os.path.join(PROJECT_PATH, 'models')


from importlib.machinery import SourceFileLoader
facenet = SourceFileLoader('facenet', os.path.join(PROJECT_PATH, 'facenet.py')).load_module()
mx2tfrecords = SourceFileLoader('mx2tfrecords', os.path.join(PROJECT_PATH, 'data/mx2tfrecords.py')).load_module()

L_Resnet_E_IR_fix_issue9 = SourceFileLoader('L_Resnet_E_IR_fix_issue9', os.path.join(PROJECT_PATH, 'nets/L_Resnet_E_IR_fix_issue9.py')).load_module()

face_losses = SourceFileLoader('face_losses', os.path.join(PROJECT_PATH, 'losses/face_losses.py')).load_module()
eval_data_reader = SourceFileLoader('eval_data_reader', os.path.join(PROJECT_PATH, 'data/eval_data_reader.py')).load_module()
verification = SourceFileLoader('verification', os.path.join(PROJECT_PATH, 'verification.py')).load_module()


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

    # insightface_dataset = eval_dir_path
    insightface_pair = os.path.join(PROJECT_PATH, 'data/First_100_ALL VIS_112_1.txt')
    insightface_dataset = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\First_100_ALL VIS_112"

    insightface_image_size = 112
    batch_size = 32

    facenet_image_size = 160
    facenet_dataset = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\First_100_ALL VIS_160"
    facenet_batch_size = batch_size
    facenet_model = os.path.join(PROJECT_PATH, 'models/facenet/20180402-114759')
    facenet_pairs = insightface_pair


def data_iter(datasets, batch_size):
    data_num = datasets.shape[0]
    for i in range(0, data_num, batch_size):
        yield datasets[i:min(i+batch_size, data_num), ...]


def get_facenet_embeddings(args):
    # Read the file containing the pairs used for testing
    # ver_list = []
    ver_name_list = []
    print('begin db %s convert.' % args.facenet_dataset)

    data_set = eval_data_reader.load_eval_datasets_2(args, facenet=True)
    # ver_list.append(data_set)
    ver_name_list.append(args.facenet_dataset)

    #  Evaluate custom dataset with facenet pre-trained model
    print("Evaluate custom dataset with facenet pre-trained model")
    with tf.Graph().as_default():
        with tf.Session() as sess:
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            image_batch = tf.placeholder(name='img_inputs', shape=[None, args.facenet_image_size, args.facenet_image_size, 3], dtype=tf.float32)
            label_batch = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int32)

            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(args.facenet_model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            # results = verification.ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess, embedding_tensor=embeddings,
            #                                 batch_size=args.batch_size, feed_dict=input_map, input_placeholder=image_batch,
            #                                 phase_train_placeholder=phase_train_placeholder)

            # results = []
            # for i in range(len(ver_list)):
            # acc1, std1, acc2, std2, xnorm, embeddings_list = test(data_set=ver_list[i], sess=sess, embedding_tensor=embedding_tensor,
            #                                                       batch_size=batch_size, feed_dict=feed_dict,
            #                                                       input_placeholder=input_placeholder,
            #                                                       phase_train_placeholder=phase_train_placeholder)

            batch_size = args.batch_size
            input_placeholder = image_batch

            # data_set = ver_list[0]
            # feed_dict = input_map
            print('testing verification..')
            data_list = data_set[0]
            issame_list = data_set[1]
            embeddings_list = []
            time_consumed = 0.0
            for i in range(len(data_list)):
                datas = data_list[i]
                embeddings_array = None
                # feed_dict.setdefault(input_placeholder, None)
                for idx, data in enumerate(data_iter(datas, batch_size)):
                    data_tmp = data.copy()  # fix issues #4
                    data_tmp -= 127.5
                    data_tmp *= 0.0078125

                    time0 = datetime.datetime.now()

                    # if phase_train_placeholder is not None:
                    mr_feed_dict = {input_placeholder: data_tmp, phase_train_placeholder: False}
                    _embeddings = sess.run(embeddings, mr_feed_dict)
                    # else:
                    #     feed_dict[input_placeholder] = data_tmp
                    #     time0 = datetime.datetime.now()
                    #     _embeddings = sess.run(embeddings, feed_dict)

                    time_now = datetime.datetime.now()
                    diff = time_now - time0
                    time_consumed += diff.total_seconds()
                    if embeddings_array is None:
                        embeddings_array = np.zeros((datas.shape[0], _embeddings.shape[1]))
                    try:
                        embeddings_array[idx * batch_size:min((idx + 1) * batch_size, datas.shape[0]), ...] = _embeddings
                    except ValueError:
                        print('idx*batch_size value is %d min((idx+1)*batch_size, datas.shape[0]) %d, batch_size %d, data.shape[0] %d' %
                              (idx * batch_size, min((idx + 1) * batch_size, datas.shape[0]), batch_size, datas.shape[0]))
                        print('embedding shape is ', _embeddings.shape)
                embeddings_list.append(embeddings_array)

            embeddings_list_copy = copy.deepcopy(embeddings_list)
            _xnorm = 0.0
            _xnorm_cnt = 0
            for embed in embeddings_list_copy:
                for i in range(embed.shape[0]):
                    _em = embed[i]
                    _norm = np.linalg.norm(_em)
                    # print(_em.shape, _norm)
                    _xnorm += _norm
                    _xnorm_cnt += 1
            _xnorm /= _xnorm_cnt

            final_embeddings_output = embeddings_list[0] + embeddings_list[1]
            # final_embeddings_output_copy = embeddings_list_copy[0] + embeddings_list_copy[1]
            final_embeddings_output = sklearn.preprocessing.normalize(final_embeddings_output)
            # final_embeddings_output_copy = sklearn.preprocessing.normalize(final_embeddings_output_copy)
            print(final_embeddings_output.shape)
            # print(final_embeddings_output_copy.shape)

            return embeddings_list, final_embeddings_output, _xnorm


def get_insightface_embeddings(args):
    # Read the file containing the pairs used for testing
    # ver_list = []
    # ver_name_list = []
    print('begin db %s convert.' % args.insightface_dataset)

    data_set = eval_data_reader.load_eval_datasets_2(args, facenet=False)
    # ver_list.append(data_set)
    # ver_name_list.append(args.insightface_dataset)

    #  Evaluate custom dataset with InsightFace_TF pre-trained model
    print("Evaluate custom dataset with InsightFace_TF pre-trained model")
    images = tf.placeholder(name='img_inputs', shape=[None, args.insightface_image_size, args.insightface_image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = L_Resnet_E_IR_fix_issue9.get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
    embeddings = net.outputs
    # mv_mean = tl.layers.get_variables_with_name('resnet_v1_50/bn0/moving_mean', False, True)[0]
    # 3.2 get arcface loss
    logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output)

    sess = tf.Session()
    saver = tf.train.Saver()

    # result_index = []
    # for file_index in args.ckpt_index_list:
    feed_dict_test = {}
    path = args.ckpt_file + args.ckpt_index_list[0]
    saver.restore(sess, path)
    print('ckpt file %s restored!' % args.ckpt_index_list[0])
    feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
    feed_dict_test[dropout_rate] = 1.0

    # results = verification.ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess, embedding_tensor=embeddings,
    #                                 batch_size=args.batch_size, feed_dict=input_map, input_placeholder=image_batch,
    #                                 phase_train_placeholder=phase_train_placeholder)

    # results = []
    # for i in range(len(ver_list)):
    # acc1, std1, acc2, std2, xnorm, embeddings_list = test(data_set=ver_list[i], sess=sess, embedding_tensor=embedding_tensor,
    #                                                       batch_size=batch_size, feed_dict=feed_dict,
    #                                                       input_placeholder=input_placeholder,
    #                                                       phase_train_placeholder=phase_train_placeholder)

    batch_size = args.batch_size
    input_placeholder = images

    # data_set = ver_list[0]
    # feed_dict = input_map
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0

    for i in range(len(data_list)):
        datas = data_list[i]
        embeddings_array = None
        feed_dict_test.setdefault(input_placeholder, None)
        for idx, data in enumerate(data_iter(datas, batch_size)):
            data_tmp = data.copy()  # fix issues #4
            data_tmp -= 127.5
            data_tmp *= 0.0078125

            # time0 = datetime.datetime.now()

            # if phase_train_placeholder is not None:
            # mr_feed_dict = {input_placeholder: data_tmp, phase_train_placeholder: False}
            # _embeddings = sess.run(embeddings, mr_feed_dict)
            # else:
            feed_dict_test[input_placeholder] = data_tmp
            time0 = datetime.datetime.now()
            _embeddings = sess.run(embeddings, feed_dict_test)

            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings_array is None:
                embeddings_array = np.zeros((datas.shape[0], _embeddings.shape[1]))
            try:
                embeddings_array[idx * batch_size:min((idx + 1) * batch_size, datas.shape[0]), ...] = _embeddings
            except ValueError:
                print('idx*batch_size value is %d min((idx+1)*batch_size, datas.shape[0]) %d, batch_size %d, data.shape[0] %d' %
                      (idx * batch_size, min((idx + 1) * batch_size, datas.shape[0]), batch_size, datas.shape[0]))
                print('embedding shape is ', _embeddings.shape)
        embeddings_list.append(embeddings_array)

    embeddings_list_copy = copy.deepcopy(embeddings_list)
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list_copy:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            # print(_em.shape, _norm)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    final_embeddings_output = embeddings_list[0] + embeddings_list[1]
    # final_embeddings_output_copy = embeddings_list_copy[0] + embeddings_list_copy[1]
    final_embeddings_output = sklearn.preprocessing.normalize(final_embeddings_output)
    # final_embeddings_output_copy = sklearn.preprocessing.normalize(final_embeddings_output_copy)
    print(final_embeddings_output.shape)
    # print(final_embeddings_output_copy.shape)

    return embeddings_list, final_embeddings_output, _xnorm

if __name__ == '__main__':
    # args = get_args()
    args = Args()
    # get_facenet_embeddings(args)
    # exit(0)

    get_insightface_embeddings(args)
    exit(0)

    # Read the file containing the pairs used for testing
    ver_list = []
    ver_name_list = []
    print('begin db %s convert.' % args.insightface_dataset)

    data_set = eval_data_reader.load_eval_datasets_2(args, facenet=False)
    ver_list.append(data_set)
    ver_name_list.append(args.insightface_dataset)

    #  -------------------------------------------------------------------------------------------------------------------
    print(f"{'='}"*40)

    #  Evaluate custom dataset with InsightFace_TF pre-trained model
    print("Evaluate custom dataset with InsightFace_TF pre-trained model")
    images = tf.placeholder(name='img_inputs', shape=[None, args.insightface_image_size, args.insightface_image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = L_Resnet_E_IR_fix_issue9.get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
    embedding_tensor = net.outputs
    # mv_mean = tl.layers.get_variables_with_name('resnet_v1_50/bn0/moving_mean', False, True)[0]
    # 3.2 get arcface loss
    logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output)

    sess = tf.Session()
    saver = tf.train.Saver()

    result_index = []
    for file_index in args.ckpt_index_list:
        feed_dict_test = {}
        path = args.ckpt_file + file_index
        saver.restore(sess, path)
        print('ckpt file %s restored!' % file_index)
        feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
        feed_dict_test[dropout_rate] = 1.0
        results = verification.ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess,
                           embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test,
                           input_placeholder=images)
        result_index.append(results)
    print(result_index)

    tf.reset_default_graph()

    #  -------------------------------------------------------------------------------------------------------------------
    print(f"{'='}"*40)
    
    # Read the file containing the pairs used for testing
    ver_list = []
    ver_name_list = []
    print('begin db %s convert.' % args.insightface_dataset)

    data_set = eval_data_reader.load_eval_datasets_2(args, facenet=True)
    ver_list.append(data_set)
    ver_name_list.append(args.insightface_dataset)
    
    #  Evaluate custom dataset with facenet pre-trained model
    print("Evaluate custom dataset with facenet pre-trained model")
    with tf.Graph().as_default():

        with tf.Session() as sess:
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            image_batch = tf.placeholder(name='img_inputs', shape=[None, args.facenet_image_size, args.facenet_image_size, 3], dtype=tf.float32)
            label_batch = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int32)

            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(args.facenet_model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            results = verification.ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess, embedding_tensor=embeddings,
                                            batch_size=args.batch_size, feed_dict=input_map, input_placeholder=image_batch,
                                            phase_train_placeholder=phase_train_placeholder)

            print(results)

