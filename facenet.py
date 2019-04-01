"""Functions for building the face recognition network.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile
import math
from six import iteritems
import pathlib
import time


def timeit(ds, BATCH_SIZE, repeat_count=1):
    from functools import reduce
    if repeat_count == -1 or repeat_count is None:
        overall_start = time.time()
        start = time.time()

        print('^^^^^^^^^^^^^^:: timeit method repeat_count ::^^^^^^^^^^^^^^')
        iterator = ds.make_initializable_iterator()
        next_element = iterator.get_next()
        sess = tf.Session()
        num_batch = 1000
        _next_element = None
        non_equal_counter = 0
        sess.run(iterator.initializer)
        for _ in range(num_batch):
            try:
                tmp = sess.run(next_element)

                if _next_element:
                    if np.count_nonzero((_next_element[0]) != (tmp[0])) == reduce(lambda x, y: x * y, _next_element[0].shape):
                        _next_element = tmp
                        non_equal_counter += 1
                else:
                    _next_element = tmp

                num_batch += 1
            except tf.errors.OutOfRangeError:
                break
                # [Perform end-of-epoch calculations here.]

        print(f'non_equal_counter: {non_equal_counter}')
        print("Num Batch: ", num_batch)

        end = time.time()

        duration = end - start
        print("{} batches: {} s".format(num_batch, duration))
        print("{:0.5f} Images/s".format(BATCH_SIZE * num_batch / duration))
        print("Total time: {}s".format(end - overall_start))

        return

    method = '2'
    if method == '1':
        overall_start = time.time()
        start = time.time()

        print('^^^^^^^^^^^^^^:: timeit method 1 ::^^^^^^^^^^^^^^')
        epoch = 20
        iterator = ds.make_initializable_iterator()
        next_element = iterator.get_next()
        sess = tf.Session()
        num_batch = 0
        _next_element = None
        non_equal_counter = 0
        for e in range(epoch):
            # print("Epoch: ", e)
            sess.run(iterator.initializer)
            while True:
                try:
                    tmp = sess.run(next_element)

                    if _next_element:
                        if np.count_nonzero((_next_element[0]) != (tmp[0])) == reduce(lambda x, y: x*y, _next_element[0].shape):
                            _next_element = tmp
                            non_equal_counter += 1
                    else:
                        _next_element = tmp

                    num_batch += 1
                except tf.errors.OutOfRangeError:
                # except Exception as err:
                #     print(err)
                #     print('OutOfRangeError')
                    break
                    # [Perform end-of-epoch calculations here.]
        print(f'non_equal_counter: {non_equal_counter}')
        print("Num Batch: ", num_batch)

        end = time.time()

        duration = end - start
        print("{} batches: {} s".format(num_batch, duration))
        print("{:0.5f} Images/s".format(BATCH_SIZE * num_batch / duration))
        print("Total time: {}s".format(end - overall_start))

    # print('->-<-'*25)

    method = '2'
    if method == '2':
        overall_start = time.time()
        start = time.time()

        print('^^^^^^^^^^^^^^:: timeit method 2 ::^^^^^^^^^^^^^^')
        epoch = 20
        ds = ds.repeat(epoch)
        iterator = ds.make_one_shot_iterator()
        next_element = iterator.get_next()
        sess = tf.Session()

        num_batch = 0
        _next_element = None
        non_equal_counter = 0

        while True:
            try:
                tmp = sess.run(next_element)

                if _next_element:
                    if np.count_nonzero((_next_element[0]) != (tmp[0])) == reduce(lambda x, y: x * y, _next_element[0].shape):
                        _next_element = tmp
                        non_equal_counter += 1
                else:
                    _next_element = tmp

                num_batch += 1
            except tf.errors.OutOfRangeError:
                # print('OutOfRangeError')
                break

        print(f'non_equal_counter: {non_equal_counter}')
        print("Num Batch: ", num_batch)

        end = time.time()

        duration = end - start
        print("{} batches: {} s".format(num_batch, duration))
        print("{:0.5f} Images/s".format(BATCH_SIZE * num_batch / duration))
        print("Total time: {}s".format(end - overall_start))


def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss


def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


def get_image_paths_and_labels(dataset, path=None):
    if path:
        name_dict = dict()
        index_dict = dict()
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        if path:
            name_dict[dataset[i].name] = str(i)
            index_dict[str(i)] = dataset[i].name

        labels_flat += [i] * len(dataset[i].image_paths)

    if path:
        return image_paths_flat, labels_flat, name_dict, index_dict

    return image_paths_flat, labels_flat


def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff


def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')


# 1: Random rotate 2: Random crop  4: Random flip  8:  Fixed image standardization  16: Flip
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16


def create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder):
    with tf.name_scope(name='Input_PipeLine'):
        images_and_labels_list = []
        for _ in range(nrof_preprocess_threads):
            filenames, label, control = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, 3)
                image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),
                                lambda:tf.py_func(random_rotate_image, [image], tf.uint8),
                                lambda:tf.identity(image))
                image = tf.cond(get_control_flag(control[0], RANDOM_CROP),
                                lambda:tf.random_crop(image, image_size + (3,)),
                                lambda:tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
                image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),
                                lambda:tf.image.random_flip_left_right(image),
                                lambda:tf.identity(image))
                image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),
                                lambda:(tf.cast(image, tf.float32) - 127.5)/128.0,
                                lambda:tf.image.per_image_standardization(image))
                image = tf.cond(get_control_flag(control[0], FLIP),
                                lambda:tf.image.flip_left_right(image),
                                lambda:tf.identity(image))
                #pylint: disable=no-member
                image.set_shape(image_size + (3,))
                images.append(image)
            images_and_labels_list.append([images, label])

        image_batch, label_batch = tf.train.batch_join(
            images_and_labels_list, batch_size=batch_size_placeholder,
            shapes=[image_size + (3,), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * 100,
            allow_smaller_final_batch=True)

        return image_batch, label_batch


def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)


def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.cast(tf.reduce_mean(x, axis=axis, keepdims=True), dtype=tf.float32)
    devs_squared = tf.square(tf.cast(x, dtype=tf.float32) - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def tf_prewhiten(x):
    mean = tf.reduce_mean(tf.cast(x, dtype=tf.float32))
    # mean, var = tf.nn.moments(x)
    std = tf.keras.backend.std(tf.cast(x, dtype=tf.float32))
    # std = reduce_std(x)
    std_adj = tf.maximum(std, 1.0/tf.sqrt(tf.cast(tf.size(x), dtype=tf.float32)))
    # y = tf.multiply(tf.subtract(x, mean), 1/std_adj)
    y = tf.multiply(tf.subtract(tf.cast(x, dtype=tf.float32), tf.cast(mean, dtype=tf.float32)), 1/std_adj)
    return y


def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True, to_gray=False):
    nrof_samples = len(image_paths)
    if to_gray:
        images = np.zeros((nrof_samples, image_size, image_size))
    else:
        images = np.zeros((nrof_samples, image_size, image_size, 3))

    for i in range(nrof_samples):
        if to_gray:
            # img = misc.imread(image_paths[i], flatten=True)
            img = misc.imread(image_paths[i], mode='L')
        else:
            img = misc.imread(image_paths[i])
            if img.ndim == 2:
                img = to_rgb(img)

        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)

        if to_gray:
            images[i, :, :] = img
        else:
            images[i,:,:,:] = img

    return images


def get_label_batch(label_data, batch_size, batch_index):
    nrof_examples = np.size(label_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = label_data[j:j+batch_size]
    else:
        x1 = label_data[j:nrof_examples]
        x2 = label_data[0:nrof_examples-j]
        batch = np.vstack([x1,x2])
    batch_int = batch.astype(np.int64)
    return batch_int


def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = image_data[j:j+batch_size,:,:,:]
    else:
        x1 = image_data[j:nrof_examples,:,:,:]
        x2 = image_data[0:nrof_examples-j,:,:,:]
        batch = np.vstack([x1,x2])
    batch_float = batch.astype(np.float32)
    return batch_float


def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                if par[1]=='-':
                    lr = -1
                else:
                    lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)


def load_and_preprocess_image(image_path, label=None, image_size=192, seed=313, normalize=True, do_resize=False, do_random_crop=False,
                                do_random_flip=False, do_prewhiten=True):
    image_string = tf.read_file(image_path)
    # img = tf.image.decode_image(image_string)

    img = tf.cond(tf.image.is_jpeg(image_string),
        lambda: tf.image.decode_jpeg(image_string, channels=3),
        lambda: tf.image.decode_png(image_string, channels=3))

    if do_prewhiten:
        img = tf_prewhiten(img)

    # https://www.tensorflow.org/api_guides/python/image
    if do_random_crop:
        # A random variable
        rand_var = tf.random_uniform([], minval=0, maxval=0.5, dtype=tf.float32, seed=seed)
        img = tf.image.central_crop(img, central_fraction=rand_var)
    if do_random_flip:
        img = tf.image.random_flip_up_down(img, seed=seed)

    # img = tf.convert_to_tensor(img, np.float32)
    if do_resize:
        img = tf.image.resize_images(img, [image_size, image_size])
    if normalize:
        tf.divide(tf.cast(img, dtype=tf.float32), tf.constant(255.0, dtype=tf.float32)) # normalize to [0,1] range
    if label is not None:
        return img, label
    return img


def preprocess_image(image, dim=96):
    # Decode it into an image tensor:
    image = tf.image.decode_image(image)
    # image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize_images(image, [dim, dim])
    # image /= 255.0  # normalize to [0,1] range

    return image


def tf_gen_dataset(image_list=None, label_list=None, nrof_preprocess_threads=4, image_size=96, method='cache', BATCH_SIZE=32,
                   seed=313, performance=False, repeat_count=1, path=None, in_memory = True, normalize=True, do_resize=False,
                   do_random_crop=False, do_random_flip=False, do_prewhiten=False):
    """

    :param image_list:
    :param label_list:
    :param nrof_preprocess_threads:
    :param image_size:
    :param method:
    :param BATCH_SIZE:
    :param seed:
    :param performance:
    :return:
    """
    if path:
        data_root = pathlib.Path(path)
        image_list = list(data_root.glob('*/*'))
        image_list = [str(path) for path in image_list]
        random.shuffle(image_list)
        image_count = len(image_list)
        print(f"image_count: {image_count}")

        # Determine the label for each image
        # List the available labels:
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        # Assign an index to each label:
        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        # Create a list of every file, and its label index
        label_list = [label_to_index[pathlib.Path(path).parent.name] for path in image_list]
        print("First 10 labels indices: ", label_list[:10])
    """
    Basic methods for training
    To train a model with this dataset you will want the data:
        - To be well shuffled.
        - To be batched.
        - To repeat forever.
        - Batches to be available as soon as possible.
    These features can be easily added using the tf.data api.
        
    There are a few things to note here:

        1. The order is important.

            A .shuffle before a .repeat would shuffle items across epoch boundaries (some items will we seen twice before others are seen at 
            all).
            A .shuffle after a .batch would shuffle the order of the batches, but not shuffle the items across batches.

        2. We use a buffer_size the same size as the dataset for a full shuffle. Up to the dataset size, large values provide better 
        randomization, but use more memory.

        3. The shuffle buffer is filled before any elements are pulled from it. So a large buffer_size may cause a delay when your Dataset is 
        starting.

        4. The shuffeled dataset doesn't report the end of a dataset until the shuffle-buffer is completely empty.
         The Dataset is restarted by .repeat, causing another wait for the shuffle-buffer to be filled.

    This last point can be addressed by using the tf.data.Dataset.apply method with the fused tf.data.experimental.shuffle_and_repeat function:
    """

    # A dataset of images
    image_count = len(image_list)
    print(f'image_count: {image_count}')
    # Using the from_tensor_slices method we can build a dataset of labels
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(label_list, tf.int64))

    if method == 'slices':
        # Note: When you have arrays like all_image_labels and all_image_paths an alternative to tf.data.dataset.Dataset.zip is to
        # slice the pair of arrays.
        ds = tf.data.Dataset.from_tensor_slices((image_list, label_list))
        image_label_ds = ds.map(lambda image_path, label: load_and_preprocess_image(image_path, label=label, image_size=image_size,
                                                                                      seed=seed, normalize=normalize, do_resize=do_resize,
                                                                                      do_random_crop=do_random_crop, do_random_flip=do_random_flip,
                                                                                      do_prewhiten=do_prewhiten),
                               num_parallel_calls=nrof_preprocess_threads)
        # Setting a shuffle buffer size as large as the dataset ensures that the data is
        # completely shuffled.
        # ds = image_label_ds.shuffle(buffer_size=image_count)
        ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count, count=repeat_count, seed=seed))
        ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=nrof_preprocess_threads)
        # ds = ds.batch(BATCH_SIZE)
        # # `prefetch` lets the dataset fetch batches, in the background while the model is training.
        # ds = ds.prefetch(buffer_size=nrof_preprocess_threads)
        if performance:
            timeit(ds, BATCH_SIZE, repeat_count)
            print('<=>' * 10)

        return ds

    elif method == 'cache_slices':
        # Cache
        # Use `tf.data.Dataset.cache` to easily cache calculations across epochs. This is especially performant if the dataq fits in memory.
        # Here the images are cached, after being pre-precessed (decoded and resized):

        # The easiest way to build a tf.data.Dataset is using the from_tensor_slices method.
        # Note: When you have arrays like all_image_labels and all_image_paths an alternative to tf.data.dataset.Dataset.zip is to
        # slice the pair of arrays.
        ds = tf.data.Dataset.from_tensor_slices((image_list, label_list))
        # Now create a new dataset that loads and formats images on the fly by mapping preprocess_image over the dataset of paths.
        image_label_ds = ds.map(lambda image_path, label: load_and_preprocess_image(image_path, label=label, image_size=image_size,
                                                                                      seed=seed, normalize=normalize, do_resize=do_resize,
                                                                                      do_random_crop=do_random_crop, do_random_flip=do_random_flip,
                                                                                      do_prewhiten=do_prewhiten),
                               num_parallel_calls=nrof_preprocess_threads)

        if in_memory:
            print('::::::::::::::::::::::::::::::::in memory cache::::::::::::::::::::::::::::::::')
            ds = image_label_ds.cache()
            ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count, count=repeat_count, seed=seed))
            ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=nrof_preprocess_threads)
            if performance:
                timeit(ds, BATCH_SIZE, repeat_count)
                print(f"{'<=>'*10}")

            # One disadvantage to using an in memory cache is that the cache must be rebuilt on each run, giving the same
            # startup delay each time the dataset is started:
        else:
            print('::::::::::::::::::::::::::::::::in disk cache::::::::::::::::::::::::::::::::')
            # If the data doesn't fit in memory, use a cache file:
            ds = image_label_ds.cache(filename='./cache.tf-data')
            ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count, count=repeat_count, seed=seed))
            ds = ds.batch(BATCH_SIZE).prefetch(nrof_preprocess_threads)
            if performance:
                timeit(ds, BATCH_SIZE, repeat_count)
                print('<=>'*10)

        return ds


def get_dataset(path, has_class_directories=True, nrof_preprocess_threads=4):

    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths


def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*(1-split_ratio)))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            nrof_images_in_class = len(paths)
            split = int(math.floor(nrof_images_in_class*(1-split_ratio)))
            if split==nrof_images_in_class:
                split = nrof_images_in_class-1
            if split>=min_nrof_images_per_class and nrof_images_in_class-split>=1:
                train_set.append(ImageClass(cls.name, paths[:split]))
                test_set.append(ImageClass(cls.name, paths[split:]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set


def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        # with gfile.FastGFile(model_exp,'rb') as f:
        # with tf.gfile.FastGFile(model_exp,'rb') as f:
        with tf.gfile.GFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
            # print()
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))

    if float(n_same) != 0.0:
        val = float(true_accept) / float(n_same)
    else:
        val = 0.0
        # print('Info: float(n_same) equal to zero')

    if float(n_diff) != 0.0:
        far = float(false_accept) / float(n_diff)
    else:
        far = 0.0
        # print('Info: float(n_diff) equal to zero')

    return val, far


def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' +  e.strerror
  
    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' +  e.strerror
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)


def list_variables(filename):
    reader = training.NewCheckpointReader(filename)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    return names


def put_images_on_grid(images, shape=(16,8)):
    nrof_images = images.shape[0]
    img_size = images.shape[1]
    bw = 3
    img = np.zeros((shape[1]*(img_size+bw)+bw, shape[0]*(img_size+bw)+bw, 3), np.float32)
    for i in range(shape[1]):
        x_start = i*(img_size+bw)+bw
        for j in range(shape[0]):
            img_index = i*shape[0]+j
            if img_index>=nrof_images:
                break
            y_start = j*(img_size+bw)+bw
            img[x_start:x_start+img_size, y_start:y_start+img_size, :] = images[img_index, :, :, :]
        if img_index>=nrof_images:
            break
    return img


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))
