# from BlindMIUtil import *
from functools import partial
import cv2 as cv
from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
from tqdm import tqdm
import argparse
import random


parser = argparse.ArgumentParser('Train and save a model (potentially with a defense')
parser.add_argument('--ndata', type=int, default=10000, help='number of data points to use')
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset to use')
parser.add_argument('--model', type=str, default='ResNet50', help='model to train as the target')
parser.add_argument('--gen', type=str, default='gaussian_noise', help='methods to generate noise')
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
# DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
# TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
TARGET_WEIGHTS_PATH = f'../models/target/{args.dataset}_{args.ndata}_{args.model}.tf'
(x_train_tar, y_train_tar), (x_test_tar, y_test_tar) = load_data(args.dataset, False, 10000)


# n_values = np.max(y_train_tar) + 1
# y_train_tar= np.eye(n_values)[y_train_tar]
# y_train_tar = y_train_tar.transpose(0, 2, 1)
# y_train_tar = y_train_tar.reshape(-1, y_train_tar.shape[1])
#
# n_test_values = np.max(y_test_tar)+1
# y_test_tar = np.eye(n_test_values)[y_test_tar]
# y_test_tar = y_test_tar.transpose(0, 2, 1)
# y_test_tar = y_test_tar.reshape(-1, y_test_tar.shape[1])



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

'''sobel method to generate artificial image data(only useful for image dataset though'''
def sobel(img_sample):
    '''literally it computes the gradients and drawing the contour'''
    img_generated = np.empty(img_sample.shape)
    for i, img in enumerate(img_sample):
        grad_x = cv.Sobel(np.float32(img), cv.CV_32F, 1, 0)
        grad_y = cv.Sobel(np.float32(img), cv.CV_32F, 0, 1)
        '''we have to use the absolute value because negative values have no meanings'''
        gradx = cv.convertScaleAbs(grad_x)
        grady = cv.convertScaleAbs(grad_y)
        '''then we bing two results together'''
        gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
        img_generated[i, :] = gradxy
    return img_generated

'''adding noise to the data'''
def gaussian_noise(img_set, mean=0, var=0.001):
    ret = np.empty(img_set.shape)
    for m, image in enumerate(img_set):
        image = np.array(image/255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        ret[m, :] = out
    return ret

'''adding salt pepper noise'''
def salt_pepper_noise(img_set, prob=0.001):
    ret = np.empty(img_set.shape)
    for m, image in enumerate(img_set):
        out = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
                rdn = random.random()
                if rdn < prob:
                    out[i] = 0
                elif rdn > thres:
                    out[i] = 255
                else:
                    out[i] = image[i]
        ret[m,:] = out
    return ret

def compute_pairwise_distances(x, y):
    '''check the shape is legal'''
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
        '''E{ K(x, x) } + E{ K(y, y) } - 2 E{ K(x, y) }'''
        cost = tf.reduce_mean(kernel(x, x))+tf.reduce_mean(kernel(y, y))-2 * tf.reduce_mean(kernel(x, y))
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost

def mmd_loss(source_samples, target_samples, weight, scope=None):
    '''sigma to chose'''
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

gen = None
print(args.gen)
if args.gen=="sobel":
    gen = sobel
if args.gen=="salt_pepper_noise":
    gen = salt_pepper_noise
if args.gen=="gaussian_noise":
    gen = gaussian_noise

def diff_Mem_attack(x_, y_true, m_true, target_model, non_Mem_Generator=gen):
    # x_ (20,000 32 32 3) y_true (20,000 100) m_true(20,000 )
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
    # print(S_target_prob_k[:5])
    nonMem_index = np.random.randint(0, x_.shape[0], size=20)
    '''select 20 random index by nonMem'''
    S_nonMem_prob_k = target_model.predict(non_Mem_Generator(x_[nonMem_index]))

    '''S_non_prob_k'''
    '''(1 for good guess)'''
    '''[1][0.77, 0.19]'''
    '''[0][0.52, 0.34]'''
    S_nonMem_prob_k = tf.convert_to_tensor(np.c_[S_nonMem_prob_k[y_true[nonMem_index].astype(bool)],
                                        np.sort(S_nonMem_prob_k, axis=1)[:, ::-1][:, :2]])

    '''S_target_prob_k'''
    '''(1 for good guess)'''
    '''[1][0.77, 0.19]     1（mem）'''
    '''[0][0.52, 0.34]     0（non）'''
    S_target_prob_k = tf.data.Dataset.from_tensor_slices((S_target_prob_k, m_true)).shuffle(buffer_size=x_.shape[0]).\
        batch(20).prefetch(tf.data.experimental.AUTOTUNE)

    m_pred, m_true = [], []
    mix_shuffled = []
    for (mix_batch, m_true_batch) in S_target_prob_k:
        m_pred_batch = np.ones(mix_batch.shape[0])
        m_pred_epoch = np.ones(mix_batch.shape[0])
        nonMemInMix = True
        while nonMemInMix:
            # print(m_pred_epoch.astype(bool))
            # print(mix_batch)
            # print(mix_batch[m_pred_epoch.astype(bool)])
            mix_epoch_new = mix_batch[m_pred_epoch.astype(bool)]
            '''compute original distance'''
            # print(S_nonMem_prob_k.shape)
            # print(mix_epoch_new.shape)
            dis_ori = mmd_loss(S_nonMem_prob_k, mix_epoch_new, weight=1)
            # print(dis_ori)
            # import sys
            # sys.exit()
            nonMemInMix = False
            for index, item in tqdm(enumerate(mix_batch)):
                '''if not in the non mem set'''
                if m_pred_batch[index] == 1:
                    '''S_non_mem union this record'''
                    nonMem_batch_new = tf.concat([S_nonMem_prob_k, [mix_batch[index]]], axis=0)
                    ''' remove unit by indexing tricks'''
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
           np.concatenate(mix_shuffled, axis=0), S_nonMem_prob_k

m_true, m_pred, mix, S_nonMem_prob_k = diff_Mem_attack(np.r_[x_train_tar, x_test_tar],
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

'''TODO: use skrlearn'''
evaluate_attack(m_true, m_pred)


















