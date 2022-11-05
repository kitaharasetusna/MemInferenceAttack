import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from dataLoader_test import *
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Input


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

'''==================================================='''
'''load target model and verify its accuracy'''
# load target model
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
# import sys
# sys.exit()
'''==================================================='''

def create_defense_classifier(input_shape, num_classes):
    inputs_b = Input(shape=input_shape)
    x_b = Dense(256, kernel_initializer=keras.initializers.glorot_uniform(seed=1000), activation='relu')(inputs_b)
    x_b = Dense(128, kernel_initializer=keras.initializers.glorot_uniform(seed=1000), activation='relu')(x_b)
    x_b = Dense(64, kernel_initializer=keras.initializers.glorot_uniform(seed=1000), activation='relu')(x_b)
    outputs_pre = Dense(num_classes, kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs = Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    model.summary()
    return model

num_class=len(y_train[0])
print('-----------------------memGuard------------------------------')

defense_model  = create_defense_classifier(input_shape=num_class, num_classes=num_class)
defense_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

'''==================================================='''
'''train defense classifier'''
x_train_attack = np.concatenate([target_model.predict(x_train), target_model.predict(x_test)],
                                 axis=0)
member_train = np.concatenate([np.ones(len(x_train)), np.zeros(len(x_test))], axis=0)

defense_model.fit(x_train_attack, member_train,
                  shuffle=True, batch_size=args.batch_size, epochs=args.epoch, verbose=1)
'''==================================================='''

'''==================================================='''
'''test defense classifier'''
x_evalute_mem, y_evalute_mem = x_1, y_1
x_evalute_non, y_evalute_non = x_4, y_4
x_evalute_attack = np.concatenate([target_model.predict(x_evalute_mem), target_model.predict(x_evalute_non)],
                                 axis=0)
member_evalute = np.concatenate([np.ones(len(x_evalute_mem)), np.zeros(len(x_evalute_non))], axis=0)
defense_model.evaluate(x=x_evalute_attack, y=member_evalute)
'''==================================================='''














