# from BlindMIUtil import *
from functools import partial
import cv2 as cv
from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser('Train and save a model (potentially with a defense')
parser.add_argument('--ndata', type=int, default=10000, help='number of data points to use')
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset to use')
parser.add_argument('--model', type=str, default='ResNet50', help='model to train as the target')
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
# DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
# TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
TARGET_WEIGHTS_PATH = f'../models/target/{args.dataset}_{args.ndata}_{args.model}.tf'
(x_train_tar, y_train_tar), (x_test_tar, y_test_tar) = load_data(args.dataset, False, 10000)
m_train = np.ones(y_train_tar.shape[0])
m_test = np.zeros(y_test_tar.shape[0])
member = np.r_[m_train, m_test]
m_true = member
# x_train_tar(10,000, 32, 32, 3)
# y_train_tar(10,000, 100)  like [ [0. 0. 1. ... 0. 0. 0.]...[]]
# np.r_[x_train_tar, x_test_tar].shape (20000, 32, 32, 3)
# np.r_[y_train_tar, y_test_tar]:  (20000, 100)
# m_true.shape:  (20000,)
'''verify the dataset is legal: '''
'''----------------------------------------------------------------------------'''
Target_Model = load_model(TARGET_WEIGHTS_PATH)
print('target model: ')
Target_Model.summary()
print('x_test_tar.shape, y_test_tar.shape: ', x_test_tar.shape, y_test_tar.shape)
print('np.r_[x_train_tar, x_test_tar]', np.r_[x_train_tar, x_test_tar].shape)
print('np.r_[y_train_tar, y_test_tar]: ', np.r_[y_train_tar, y_test_tar].shape)
print('m_true.shape: ', m_true.shape)
print('diff_Mem_attack(np.r_[x_train_tar, x_test_tar],np.r_[y_train_tar, y_test_tar],m_true, Target_Model)')
'''----------------------------------------------------------------------------'''
'''verify the dataset is legal: '''
def sobel(img_set):
    ret = np.empty(img_set.shape)
    for i, img in enumerate(img_set):
        grad_x = cv.Sobel(np.float32(img), cv.CV_32F, 1, 0)
        grad_y = cv.Sobel(np.float32(img), cv.CV_32F, 0, 1)
        gradx = cv.convertScaleAbs(grad_x)
        grady = cv.convertScaleAbs(grad_y)
        gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
        ret[i, :] = gradxy
    return ret

def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)

    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))

        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost

def mmd_loss(source_samples, target_samples, weight, scope=None):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    loss_value = maximum_mean_discrepancy(
        source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value) * weight

    return loss_value

def diff_Mem_attack(x_, y_true, m_true, target_model, non_Mem_Generator=sobel):
    y_pred = target_model.predict(x_, verbose=0)
    print('y_pred: ', y_pred.shape)
    # y_pred:  (20000, 100)
    print('y_true: ', y_true.shape)
    print('y_pre[y_true.astype(bool)]: ', y_pred[y_true.astype(bool)].shape)
    # y_pre[y_true.astype(bool)]: (20000, ) 1 if same else 0

    '''row1 y_pred[y_true.astype(bool)] [0,1...] for right guess and bad guess'''
    '''row2 and row3  biggest 2 scores <-[:, ::-1]reverse the sorted nd-array'''
    '''in this case k=2'''
    S_target_prob_k = np.c_[y_pred[y_true.astype(bool)], np.sort(y_pred, axis=1)[:, ::-1][:, :2]]
    nonMem_index = np.random.randint(0, x_.shape[0], size=20)
    '''select 20 random index by nonMem'''
    nonMem_pred = target_model.predict(non_Mem_Generator(x_[nonMem_index]))
    nonMem = tf.convert_to_tensor(np.c_[nonMem_pred[y_true[nonMem_index].astype(bool)],
                                        np.sort(nonMem_pred, axis=1)[:, ::-1][:, :2]])

    data = tf.data.Dataset.from_tensor_slices((S_target_prob_k, m_true)).shuffle(buffer_size=x_.shape[0]).\
        batch(20).prefetch(tf.data.experimental.AUTOTUNE)
    print(data)
    m_pred, m_true = [], []
    mix_shuffled = []
    for (mix_batch, m_true_batch) in data:
        m_pred_batch = np.ones(mix_batch.shape[0])
        m_pred_epoch = np.ones(mix_batch.shape[0])
        nonMemInMix = True
        while nonMemInMix:
            mix_epoch_new = mix_batch[m_pred_epoch.astype(bool)]
            dis_ori = mmd_loss(nonMem, mix_epoch_new, weight=1)
            nonMemInMix = False
            for index, item in tqdm(enumerate(mix_batch)):
                if m_pred_batch[index] == 1:
                    nonMem_batch_new = tf.concat([nonMem, [mix_batch[index]]], axis=0)
                    mix_batch_new = tf.concat([mix_batch[:index], mix_batch[index+1:]], axis=0)
                    m_pred_without = np.r_[m_pred_batch[:index], m_pred_batch[index+1:]]
                    mix_batch_new = mix_batch_new[m_pred_without.astype(bool, copy=True)]
                    dis_new = mmd_loss(nonMem_batch_new, mix_batch_new, weight=1)
                    if dis_new > dis_ori:
                        nonMemInMix = True
                        m_pred_epoch[index] = 0
            m_pred_batch = m_pred_epoch.copy()

        mix_shuffled.append(mix_batch)
        m_pred.append(m_pred_batch)
        m_true.append(m_true_batch)
    return np.concatenate(m_true, axis=0), np.concatenate(m_pred, axis=0), \
           np.concatenate(mix_shuffled, axis=0), nonMem

m_true, m_pred, mix, nonMem = diff_Mem_attack(np.r_[x_train_tar, x_test_tar],
                                              np.r_[y_train_tar, y_test_tar],
                                              m_true, Target_Model)

def evaluate_attack(m_true, m_pred):
    accuracy = tf.keras.metrics.Accuracy()
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    accuracy.update_state(m_true, m_pred)
    precision.update_state(m_true, m_pred)
    recall.update_state(m_true, m_pred)
    F1_Score = 2 * (precision.result() * recall.result()) / (precision.result() + recall.result())
    print('accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f'
          % (accuracy.result(), precision.result(), recall.result(), F1_Score))

evaluate_attack(m_true, m_pred)


















