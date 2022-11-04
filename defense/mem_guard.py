import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from dataLoader import *


parser = argparse.ArgumentParser('Train and save a model (potentially with a defense')
parser.add_argument('--ndata', type=int, default=10000, help='number of data points to use')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
parser.add_argument('--model', type=str, default='cnn', help='model to train as the target')
parser.add_argument('--epoch', type=int, default='50', help='epoch for training target/shadow models.')
parser.add_argument('--batch_size', default=10, type=int, help='batch size for training target/shadow models.')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for training target/shadow models.')
# parser.add_argument('--shadow', action='store_true', help='Train a shadow model instead of target')
args = parser.parse_args()

print(args)
def create_defense_classifier(input_shape, num_classes):
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
print('-----------------------memGuard------------------------------')



