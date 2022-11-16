import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from dataLoader_test import *
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras import backend as K
from scipy.special import softmax

parser = argparse.ArgumentParser('Train and save a model (potentially with a defense')
parser.add_argument('--ndata', type=int, default=10000, help='number of data points to use')
parser.add_argument('--dataset', type=str, default='Location', help='dataset to use')
parser.add_argument('--model', type=str, default='nn_location', help='model to train as the target')
parser.add_argument('--epoch', type=int, default='400', help='epoch for training target/shadow models.')
parser.add_argument('--batch_size', default=100, type=int, help='batch size for training target/shadow models.')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for training target/shadow models.')
parser.add_argument('--shadow', action='store_true', help='Train a shadow model instead of target')
args = parser.parse_args()
num_classes = 30
args.shadow = False

print(args)
user_label_dim = num_classes
'''==================================================='''
'''load target model and verify its accuracy'''
target_model_path = f'../models/target/{args.dataset}_{args.ndata}_{args.model}.tf'
target_model = tf.keras.models.load_model(target_model_path)
target_model.summary()
(x_1, y_1), (x_2_m, y_2_m), (x_2_n, y_2_n), (x_3, y_3), (x_4, y_4)= load_Location(args.shadow, args.ndata)
x_train, y_train = x_1, y_1
x_test, y_test = x_3, y_3
y_train, y_test = y_train.astype(int), y_test.astype(int)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
target_model.evaluate(x=x_train, y=y_train)
target_model.evaluate(x=x_test, y=y_test)
'''==================================================='''

'''==================================================='''
'''load defense model and verify its accuracy'''
model_path = f'../models/target/{args.dataset}_{args.ndata}_{args.model}_defense.tf'
defense_model = tf.keras.models.load_model(model_path)
defense_model.summary()
defense_model.trainable = False

x_train_attack = np.concatenate([target_model.predict(x_train), target_model.predict(x_test)],
                                 axis=0)
member_train = np.concatenate([np.ones(len(x_train)), np.zeros(len(x_test))], axis=0)
x_evalute_mem, y_evalute_mem = x_1, y_1
x_evalute_non, y_evalute_non = x_4, y_4
x_evalute_attack = np.concatenate([target_model.predict(x_evalute_mem), target_model.predict(x_evalute_non)],
                                 axis=0)
member_evalute = np.concatenate([np.ones(len(x_evalute_mem)), np.zeros(len(x_evalute_non))], axis=0)
defense_model.evaluate(x=x_train_attack, y=member_train)
defense_model.evaluate(x=x_evalute_attack, y=member_evalute)

'''=========================================================================================='''

'''compute s(true confidence score vector) and z(logits of the target classifier)'''
'''s = f_evaluate; z = f_evalute_logits'''
sub_model = tf.keras.models.Model(inputs = target_model.input,
                                  outputs = target_model.layers[-2].output)
x_evalute_input = np.concatenate([x_evalute_mem, x_evalute_non],
                                 axis=0)
f_evaluate = target_model.predict(x_evalute_input)
f_evaluate_logits = sub_model.predict(x_evalute_input)
# target_model.summary()
# print(f_evaluate.shape)
# print(np.sum(f_evaluate, axis=1))
# print(f_evaluate_logits.shape)
# print(np.sum(f_evaluate_logits, axis=1))

del target_model

'''keep a copy since we have to sort the s and z'''
'''------------------------------------origin-------------------------------'''
f_evaluate_origin = np.copy(f_evaluate)  #keep a copy of original one
f_evaluate_logits_origin = np.copy(f_evaluate_logits)

sort_index = np.argsort(f_evaluate, axis=1)
'''------------------------------------origin-------------------------------'''
# print(sort_index)

# import sys
# sys.exit()

'''order of each record'''
back_index = np.copy(sort_index)
'''order of each record'''

for i in np.arange(back_index.shape[0]):
    back_index[i, sort_index[i, :]] = np.arange(back_index.shape[1])
f_evaluate = np.sort(f_evaluate, axis=1)
f_evaluate_logits = np.sort(f_evaluate_logits, axis=1)
'''----------------------------------------------------------------------'''

'''compute the h(logits of defender's defense classifier)'''
'''h  '''

sub_model_defense = tf.keras.models.Model(inputs = defense_model.input,
                                  outputs = defense_model.layers[-2].output[:, 0])

f_evaluate_logits_softmax = tf.nn.softmax(f_evaluate_logits_origin, axis=1)
# print(np.sum(f_evaluate_logits_softmax, axis=1))
h = sub_model_defense.predict(f_evaluate_logits_softmax)
# print(h)
'''----------------------------------------h-------------------------------'''


# print(tf.nn.softmax(h, axis=1))
c1 = 1.0  #used to find adversarial examples
c2 = 10.0    #penalty such that the index of max score is keeped
c3 = 0.1




success_fraction = 0.0
max_iteration = 300
result_array = np.zeros(f_evaluate.shape,dtype=float)
result_array_logits = np.zeros(f_evaluate.shape,dtype=float)
np.random.seed(1000)
'''---------------------------------------- getting z+e(logits of the target classifier) -------------------------------'''
for test_sample_id in np.arange(0,f_evaluate_origin.shape[0]):
    if test_sample_id%100==0:
        print("test sample id: {}".format(test_sample_id))
    max_label = np.argmax(f_evaluate_origin[test_sample_id, :])
    print('label is: ', max_label)
    # print(f_evaluate_origin[test_sample_id, :])
    origin_value = np.copy(f_evaluate_origin[test_sample_id, :]).reshape(1, user_label_dim)
    # print(origin_value)
    # print(np.sum(origin_value, axis=1))
    origin_value_logits = np.copy(f_evaluate_logits_origin[test_sample_id,
                                  :]).reshape(1, user_label_dim)
    label_mask_array = np.zeros([1, user_label_dim], dtype=float)
    label_mask_array[0, :] = 0.0
    label_mask_array[0, max_label] = 1.0
    label_mask_array = tf.convert_to_tensor(label_mask_array, dtype=tf.float32)
    # print(label_mask_array)
    # print(origin_value_logits)
    # print(np.sum(origin_value_logits, axis=1))
    sample_f = np.copy(origin_value_logits)
    result_predict_scores_initial = defense_model.predict(tf.nn.softmax(sample_f))
    # print(result_predict_scores_initial)

    '''if the original guess of defense classifier is already bad, do nothings'''
    if np.abs(result_predict_scores_initial - 0.5) <= 1e-5:
        success_fraction += 1.0
        result_array[test_sample_id, :] = origin_value[0, back_index[test_sample_id, :]]
        result_array_logits[test_sample_id, :] \
            = origin_value_logits[0, back_index[test_sample_id, :]]
        continue
    '''s_prime'''
    last_iteration_result \
        = np.copy(origin_value)[0, :]
    '''z+e'''
    last_iteration_result_logits \
        = np.copy(origin_value_logits)[0, :]

    success = True
    c3 = 0.1
    iterate_time = 1
    while success == True:
        sample_f = np.copy(origin_value_logits)
        j = 1
        result_max_label = -1
        result_predict_scores = result_predict_scores_initial
        while j < max_iteration and (max_label != result_max_label
                                     or (result_predict_scores - 0.5)
                                     *(result_predict_scores_initial - 0.5) > 0):
            # print(j < max_iteration and (max_label != result_max_label or (result_predict_scores - 0.5) *(result_predict_scores_initial - 0.5) > 0))
            # print(max_label!=result_max_label)
            # print(max_label!=result_max_label or (result_predict_scores - 0.5)*(result_predict_scores_initial-0.5)>0)
            # print(result_max_label, max_label)
            # print(result_predict_scores, result_predict_scores_initial)
            sample_f_watch = tf.Variable(sample_f, trainable=True)
            # if j == 2:
            #     import  sys
            #     sys.exit()
            '''to auto differentiate loss by sample_f using tf.GradientTape() in tf2'''
            with tf.GradientTape() as g:
                g.watch(sample_f_watch)
                # print(sample_f_watch)
                '''compute loss1'''
                f_evaluate_logits_softmax = tf.nn.softmax(sample_f_watch, axis=1)
                # print(f_evaluate_logits_softmax)
                h = sub_model_defense(f_evaluate_logits_softmax)
                # print(h)
                loss1 = tf.abs(h)
                # print(loss1)
                # print(g.gradient(loss, sample_f_watch))
                # print(sample_f_watch)
                '''compute loss2'''
                correct_label = tf.reduce_sum(label_mask_array * sample_f_watch, axis=1)
                # print(correct_label)
                wrong_label = tf.reduce_max((1 - label_mask_array) * sample_f_watch - 1e8 * label_mask_array, axis=1)
                loss2 = tf.nn.relu(wrong_label - correct_label)
                # print(wrong_label)
                # print(loss2)

                # print(g.gradient(loss, sample_f_watch))
                '''compute loss3'''
                loss3 = tf.reduce_sum(tf.abs(tf.nn.softmax(sample_f_watch) - origin_value))  # L-1 norm
                # print(loss3)
                loss = c1 * loss1 + c2 * loss2 + c3 * loss3
            gradient_values = g.gradient(loss, sample_f_watch)
            # print(gradient_values)
            # print()
            gradient_values_norm = tf.linalg.norm(gradient_values, ord='euclidean',
                                           axis=None, keepdims=None, name=None)
            # print(gradient_values_norm)
            gradient_values = tf.math.divide(gradient_values, gradient_values_norm)
            # print(gradient_values)
            # print(tf.math.reduce_sum(gradient_values, axis=1))
            gradient_values_ndarry = tf.make_ndarray(tf.make_tensor_proto(gradient_values))
            # print(sample_f)
            sample_f = sample_f-0.1*gradient_values_ndarry
            # print(gradient_values)
            # print(sample_f)
            result_predict_scores = defense_model.predict(tf.nn.softmax(sample_f))
            # print(result_predict_scores)
            result_max_label = np.argmax(sample_f)
            # print(max_label)
            # print(result_max_label)
            j+=1
        # print(max_label, result_max_label)
        # print(max_label!=result_max_label)
        # print(defense_model.predict(sample_f)-0.5)
        # print(result_predict_scores_initial-0.5)
        # print(((defense_model.predict(sample_f)-0.5)*(result_predict_scores_initial-0.5))>0)

        if max_label!=result_max_label:
            if iterate_time==1:
                print("failed sample for label not same for id: {},"
                      "c3:{} not add noise".format(test_sample_id,c3))
                success_fraction-=1.0
            break
        if ((defense_model.predict(sample_f)-0.5)*(result_predict_scores_initial-0.5))>0:
            if iterate_time==1:
                print("max iteration reached with id: {}, "
                      "max score: {}, prediction_score: {}, "
                      "c3: {}, not add noise".format(test_sample_id,
                                                     defense_model.predict(sample_f),
                                                     result_predict_scores, c3))
            break
        last_iteration_result[:] = softmax(sample_f)[0, :]
        last_iteration_result_logits[:] = sample_f[0, :]
        iterate_time += 1
        c3 = c3 * 10
        print(c3)
        if c3 > 100000:
            break
    success_fraction += 1.0
    result_array[test_sample_id, :] = last_iteration_result[:]
    result_array_logits[test_sample_id, :] = last_iteration_result_logits[:]
# import sys
# sys.exit()
print("Success fraction: {}".format(success_fraction / float(f_evaluate.shape[0])))
# import sys
# sys.exit()

np.savez("noise_data_{}.npz".format(args.qt),
         defense_output=result_array,
         defense_output_logits=result_array_logits,
         tc_output=f_evaluate_origin,
         tc_output_logits=f_evaluate_logits_origin,
         predict_origin=predict_origin
         ,predict_modified=predict_modified)









































































