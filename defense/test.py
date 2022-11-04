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
parser.add_argument('--batch_size', default=10, type=int, help='batch size for training target/shadow models.')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for training target/shadow models.')
# parser.add_argument('--shadow', action='store_true', help='Train a shadow model instead of target')
args = parser.parse_args()

target_model_path = f'../models/target/{args.dataset}_{args.ndata}_{args.model}.tf'
target_model = tf.keras.models.load_model(target_model_path)

print(target_model.summary())

