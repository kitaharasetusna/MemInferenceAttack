import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from dataLoader import *
from ModelUtil import *
import argparse

parser = argparse.ArgumentParser('Train and save a model (potentially with a defense')
parser.add_argument('--ndata', type=int, default=10000, help='number of data points to use')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
parser.add_argument('--model', type=str, default='cnn', help='model to train as the target')
parser.add_argument('--epoch', type=int, default='50', help='epoch for training target/shadow models.')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for training target/shadow models.')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for training target/shadow models.')
parser.add_argument('--shadow', action='store_true', help='Train a shadow model instead of target')
args = parser.parse_args()

args.shadow = False
print(args)
if not args.shadow:
    model_path = f'models/target/{args.dataset}_{args.ndata}_{args.model}.tf'
else:
    model_path = f'models/shadow/{args.dataset}_{args.ndata}_{args.model}.tf'

print(model_path)
(x_train, y_train), (x_test, y_test) = load_data(args.dataset, args.shadow, args.ndata)

print(x_train.shape[1:])
# import sys
# sys.exit()
print(x_train.shape)
model = globals()['create_{}_model'.format(args.model)](x_train.shape[1:], y_train.shape[1])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

model.fit(x_train,
          y_train,
          validation_data=(x_test,y_test),
          batch_size=args.batch_size,
          epochs=args.epoch)

model.save(model_path)
