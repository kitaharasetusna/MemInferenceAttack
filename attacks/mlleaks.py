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
if args.dataset=='cifar10':
    num_classes = 10
else:
    num_classes = 100
def create_attack_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Dense(128, input_shape=input_shape, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model

def clipDataTopX(dataToClip, top=3):
    res = [sorted(s, reverse=True)[0:top] for s in dataToClip]
    return np.array(res)

# load data
(x_train, y_train), (x_shadow, y_shadow) = load_data(args.dataset, False, args.ndata)
num_class=len(y_train[0])
print('-----------------------mlleaks------------------------------')
attack_model= create_attack_model(input_shape=(3,),num_classes=2)
attack_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

# load target and shadow model
target_model_path = f'../models/target/{args.dataset}_{args.ndata}_{args.model}.tf'
shadow_model_path = f'../models/shadow/{args.dataset}_{args.ndata}_{args.model}.tf'
target_model = tf.keras.models.load_model(target_model_path)
shadow_model = tf.keras.models.load_model(shadow_model_path)

# prepare attack model's training data and train attack model
x_attack = np.concatenate([shadow_model.predict(x_shadow), shadow_model.predict(x_train)], axis=0)
x_attack = clipDataTopX(x_attack, top=3)
member = np.concatenate([np.ones(len(x_shadow)), np.zeros(len(x_train))], axis=0)
print(member)
print(x_attack.shape, x_attack.shape, member.shape)
attack_model.fit(x_attack, member,
                shuffle=True, batch_size=args.batch_size, epochs=args.epoch, verbose=1)
# print(f"train MI: {attack_model.evaluate(x_attack, member, verbose=2)}")

# test attack model
x_test_attack = np.concatenate([target_model.predict(x_train), target_model.predict(x_shadow)],
                                 axis=0)
x_test_attack = clipDataTopX(x_test_attack, top=3)
member_test = np.concatenate([np.ones(len(x_train)), np.zeros(len(x_shadow))], axis=0)
print(f"test MI: {attack_model.evaluate(x_test_attack, member_test, verbose=2)}")
result=[1 if m[1] > 0.5 else 0 for m in attack_model.predict(x_test_attack)]
print(classification_report(member_test, result, digits=4))