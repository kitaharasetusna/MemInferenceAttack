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
from sklearn.metrics import classification_report, accuracy_score, precision_score,roc_curve

parser = argparse.ArgumentParser('Train and save a model (potentially with a defense')
parser.add_argument('--ndata', type=int, default=10000, help='number of data points to use')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
parser.add_argument('--model', type=str, default='cnn', help='model to train as the target')
parser.add_argument('--epoch', type=int, default='50', help='epoch for training target/shadow models.')
parser.add_argument('--batch_size', default=100, type=int, help='batch size for training target/shadow models.')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for training target/shadow models.')
# parser.add_argument('--shadow', action='store_true', help='Train a shadow model instead of target')
args = parser.parse_args()

print(args)
def create_attack_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Dense(128, input_shape=input_shape, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model



# load data
(x_train, y_train), (x_shadow, y_shadow) = load_data(args.dataset, False, args.ndata)

print('-----------------------loss based------------------------------')
attack_model= create_attack_model(input_shape=(3,),num_classes=2)
attack_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

# load target and shadow model
target_model_path = f'models/target/{args.dataset}_{args.ndata}_{args.model}.tf'
shadow_model_path = f'models/shadow/{args.dataset}_{args.ndata}_{args.model}.tf'
target_model = tf.keras.models.load_model(target_model_path)
shadow_model = tf.keras.models.load_model(shadow_model_path)

# get threshold from shadow model
x_attack = np.concatenate([shadow_model.predict(x_shadow), shadow_model.predict(x_train)], axis=0)
y_attack = np.concatenate([y_shadow, y_train], axis=0)
member_roc = np.concatenate([np.zeros(len(x_shadow)), np.ones(len(x_train))], axis=0)
member = np.concatenate([np.ones(len(x_shadow)), np.zeros(len(x_train))], axis=0)
loss_attack = (tf.keras.losses.categorical_crossentropy(y_attack,x_attack)).numpy()


accs=[]
pres=[]
fpr, tpr, threshs = roc_curve(member_roc,loss_attack)
for thresh in threshs:
    accs.append(accuracy_score(member,[1 if m < thresh else 0 for m in loss_attack]))
    pres.append(precision_score(member, [1 if m < thresh else 0 for m in loss_attack]))

accs, pres = np.array(accs), np.array(pres)
print(accs.max())
max_thresh = threshs[accs.argmax()]


# print(f"train MI: {attack_model.evaluate(x_attack, member, verbose=2)}")

# test attack model
x_test_attack = np.concatenate([target_model.predict(x_train), target_model.predict(x_shadow)],
                                 axis=0)
y_test_attack = np.concatenate([y_train, y_shadow], axis=0)
loss_test_attack = tf.keras.losses.categorical_crossentropy(y_test_attack,x_test_attack).numpy()
member_test_roc = np.concatenate([np.zeros(len(x_train)), np.ones(len(x_shadow))], axis=0)
member_test = np.concatenate([np.ones(len(x_train)), np.zeros(len(x_shadow))], axis=0)

result = [1 if m < max_thresh else 0 for m in loss_test_attack]

print(classification_report(member_test, result, digits=4))