import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import metrics
import tensorflow as tf
from ModelUtil import *
import argparse
import numpy as np
from dataLoader import *
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser('Train and save a model (potentially with a defense')
parser.add_argument('--ndata', type=int, default=10000, help='number of data points to use')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
parser.add_argument('--model', type=str, default='cnn', help='model to train as the target')
parser.add_argument('--epoch', type=int, default='50', help='epoch for training target/shadow models.')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for training target/shadow models.')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for training target/shadow models.')
# parser.add_argument('--shadow', action='store_true', help='Train a shadow model instead of target')
args = parser.parse_args()

print(args)
def create_attack_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Dense(64, input_shape=input_shape, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model

# load data
(x_train, y_train), (x_shadow, y_shadow) = load_data(args.dataset, False, args.ndata)
num_class=len(y_train[0])
print('-----------------------mia17------------------------------')
attack_models=[]
for i in range(num_class):
    attack_model=create_attack_model(input_shape=(num_class*2,),num_classes=2)
    attack_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                         metrics=['accuracy'])
    attack_models.append(attack_model)

# load target and shadow model
target_model_path = f'../models/target/{args.dataset}_{args.ndata}_{args.model}.tf'
shadow_model_path = f'../models/shadow/{args.dataset}_{args.ndata}_{args.model}.tf'
target_model = tf.keras.models.load_model(target_model_path)
shadow_model = tf.keras.models.load_model(shadow_model_path)

# prepare attack model's training data and train attack model
# all training data
x_attack = np.concatenate([shadow_model.predict(x_shadow), shadow_model.predict(x_train)], axis=0)
y_attack = np.concatenate([y_shadow, y_train], axis=0)
x_attack = np.concatenate([x_attack,y_attack], axis=1)
member = np.concatenate([np.ones(len(x_shadow)), np.zeros(len(x_train))], axis=0)
# divide by class
x_attacks=[]
members=[]
for i in range(num_class):
    x_attacks.append(x_attack[x_attack[:,i+num_class]==1])
    members.append(member[x_attack[:,i+num_class]==1])

x_attacks, members = np.array(x_attacks), np.array(members)
print(x_attack.shape, x_attack.shape, member.shape)

# train models by class
for i in range(num_class):
    attack_models[i].fit(x_attacks[i], members[i],
                    shuffle=True, batch_size=args.batch_size, epochs=args.epoch, verbose=1)
# print(f"train MI: {attack_model.evaluate(x_attack, member, verbose=2)}")

# test attack model
x_test_attack = np.concatenate([target_model.predict(x_train), target_model.predict(x_shadow)],
                                 axis=0)
y_test_attack = np.concatenate([y_train, y_shadow], axis=0)
x_test_attack = np.concatenate([x_test_attack,y_test_attack], axis=1)
member_test = np.concatenate([np.ones(len(x_train)), np.zeros(len(x_shadow))], axis=0)

x_test_attacks=[]
member_tests=[]
members=[]
for i in range(num_class):
    x_test_attacks.append(x_test_attack[x_test_attack[:,i+num_class]==1])
    member_tests.append(member_test[x_test_attack[:,i+num_class]==1])
    members.extend(member_test[x_test_attack[:,i+num_class]==1])

x_test_attacks, member_tests = np.array(x_test_attacks), np.array(member_tests)
results=[]
for i in range(num_class):
    result=[1 if m[1] > 0.5 else 0 for m in attack_models[i].predict(x_test_attacks[i])]
    results.extend(result)

print(classification_report(members, results, digits=4))