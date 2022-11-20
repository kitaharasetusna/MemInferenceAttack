from BlindMIUtil import *

from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
TARGET_WEIGHTS_PATH = f'../models/target/{DATA_NAME}_{10000}_{TARGET_MODEL_GENRE}.tf'
(x_train_tar, y_train_tar), (x_test_tar, y_test_tar) = load_data(DATA_NAME, False, 10000)
m_train = np.ones(y_train_tar.shape[0])
m_test = np.zeros(y_test_tar.shape[0])
member = np.r_[m_train, m_test]
m_true = member
# x_train_tar(10,000, 32, 32, 3)
# y_train_tar(10,000, 100)  like [ [0. 0. 1. ... 0. 0. 0.]...[]]
# np.r_[x_train_tar, x_test_tar].shape (20000, 32, 32, 3)
# np.r_[y_train_tar, y_test_tar]:  (20000, 100)
# m_true.shape:  (20000,)

Target_Model = load_model(TARGET_WEIGHTS_PATH)
print('target model: ')
Target_Model.summary()
print('x_test_tar.shape, y_test_tar.shape: ', x_test_tar.shape, y_test_tar.shape)
print('np.r_[x_train_tar, x_test_tar]', np.r_[x_train_tar, x_test_tar].shape)
print('np.r_[y_train_tar, y_test_tar]: ', np.r_[y_train_tar, y_test_tar].shape)
print('m_true.shape: ', m_true.shape)
print('diff_Mem_attack(np.r_[x_train_tar, x_test_tar],np.r_[y_train_tar, y_test_tar],m_true, Target_Model)')

def diff_Mem_attack(x_, y_true, m_true, target_model, non_Mem_Generator=sobel):
    # cifar
    # x_ (20000, 32, 32, 3)
    # y_true (20000, 100)
    # m_true (20000,)
    '''
    :param target_model: Fm in the paper
    :param x_: target data
    :param y_true: true labels (20000, 100)
    :param m_true: m_true (20000,) [0,1...]
    :param non_Mem_Generator: method to generate the fake data. Defaulut: Sobel.
    :return:  Tensor arrays of results
    '''

    y_pred = target_model.predict(x_, verbose=0)
    print('y_pred: ', y_pred.shape)
    # y_pred:  (20000, 100)
    print('y_true: ', y_true.shape)
    print('y_pre[y_true.astype(bool)]: ', y_pred[y_true.astype(bool)].shape)
    # y_pre[y_true.astype(bool)]: (20000, ) 1 if same else 0

    '''y_pred[y_true.astype(bool)] [0,1...] for right guess and bad guess'''
    mix = np.c_[y_pred[y_true.astype(bool)], np.sort(y_pred, axis=1)[:, ::-1][:, :2]]
    nonMem_index = np.random.randint(0, x_.shape[0], size=20)
    nonMem_pred = target_model.predict(non_Mem_Generator(x_[nonMem_index]))
    nonMem = tf.convert_to_tensor(np.c_[nonMem_pred[y_true[nonMem_index].astype(bool)],
                                        np.sort(nonMem_pred, axis=1)[:, ::-1][:, :2]])

    data = tf.data.Dataset.from_tensor_slices((mix, m_true)).shuffle(buffer_size=x_.shape[0]).\
        batch(20).prefetch(tf.data.experimental.AUTOTUNE)

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

evaluate_attack(m_true, m_pred)


















