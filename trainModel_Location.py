import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from dataLoader_test import *
from ModelUtil_test import *
import argparse
import keras

parser = argparse.ArgumentParser('Train and save a model (potentially with a defense')
parser.add_argument('--ndata', type=int, default=10000, help='number of data points to use')
parser.add_argument('--dataset', type=str, default='Location', help='dataset to use')
parser.add_argument('--model', type=str, default='nn_location', help='model to train as the target')
parser.add_argument('--epoch', type=int, default='50', help='epoch for training target/shadow models.')
parser.add_argument('--batch_size', default=100, type=int, help='batch size for training target/shadow models.')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for training target/shadow models.')
parser.add_argument('--shadow', action='store_true', help='Train a shadow model instead of target')
args = parser.parse_args()
num_classes = 30
args.shadow = False
print(args)
if not args.shadow:
    model_path = f'models/target/{args.dataset}_{args.ndata}_{args.model}.tf'
else:
    model_path = f'models/shadow/{args.dataset}_{args.ndata}_{args.model}.tf'

print(model_path)
(x_train, y_train), (x_test, y_test) = load_data(args.dataset, args.shadow, args.ndata)
'''==================================================='''
'''This part is used to encode labels from logits'''
y_train = y_train.astype(int)
y_test = y_test.astype(int)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''==================================================='''

'''==================================================='''
model = globals()['create_{}_model'.format(args.model)](x_train.shape[1:], y_train.shape[1])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
'''==================================================='''
# import sys
# sys.exit()
model.fit(x_train,
          y_train,
          validation_data=(x_test,y_test),
          batch_size=args.batch_size,
          epochs=args.epoch)

model.save(model_path)
