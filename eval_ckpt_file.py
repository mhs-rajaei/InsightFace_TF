import tensorflow as tf
import argparse
# from data.eval_data_reader import load_bin
from losses.face_losses import arcface_loss
# from nets.L_Resnet_E_IR import get_resnet
import tensorlayer as tl
from verification import ver_test
import os
from os.path import join


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

log_path = os.path.join(PROJECT_PATH, 'output')
models_path = os.path.join(PROJECT_PATH, 'models')
# train_dataset_path = r'F:\Documents\JetBrains\PyCharm\OFR\images\1024First_lfw_160'
# eval_pairs_path = os.path.join(PROJECT_PATH, 'data/All_VIS_112_pairs_3.txt')


from importlib.machinery import SourceFileLoader
facenet = SourceFileLoader('facenet', os.path.join(PROJECT_PATH, 'facenet.py')).load_module()
mx2tfrecords = SourceFileLoader('mx2tfrecords', os.path.join(PROJECT_PATH, 'data/mx2tfrecords.py')).load_module()

L_Resnet_E_IR_fix_issue9 = SourceFileLoader('L_Resnet_E_IR_fix_issue9', os.path.join(PROJECT_PATH, 'nets/L_Resnet_E_IR_fix_issue9.py')).load_module()

face_losses = SourceFileLoader('face_losses', os.path.join(PROJECT_PATH, 'losses/face_losses.py')).load_module()
eval_data_reader = SourceFileLoader('eval_data_reader', os.path.join(PROJECT_PATH, 'data/eval_data_reader.py')).load_module()
verification = SourceFileLoader('verification', os.path.join(PROJECT_PATH, 'verification.py')).load_module()
lfw = SourceFileLoader('lfw', os.path.join(PROJECT_PATH, 'lfw.py')).load_module()


class Args:
    net_depth = 50
    epoch = 1000
    batch_size = 32
    lr_steps = [40000, 60000, 80000]
    momentum = 0.9
    weight_decay = 5e-4

    image_size = [112, 112]
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

    # eval_dataset = eval_dir_path
    eval_pair = os.path.join(PROJECT_PATH, 'data/All_NIR_112_pairs_3.txt')
    eval_dataset = r'E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\All NIR_112'


if __name__ == '__main__':
    # args = get_args()
    args = Args()

    ver_list = []
    ver_name_list = []
    print('begin db %s convert.' % args.eval_dataset)

    data_set = eval_data_reader.load_eval_datasets_2(args)
    ver_list.append(data_set)
    ver_name_list.append(args.eval_dataset)

    images = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
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
        results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess,
                           embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test,
                           input_placeholder=images)
        result_index.append(results)
    print(result_index)

