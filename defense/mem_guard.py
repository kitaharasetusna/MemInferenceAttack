import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from dataLoader import *
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Input


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
    inputs_b = Input(shape=input_shape)
    x_b = Dense(256, kernel_initializer=keras.initializers.glorot_uniform(seed=1000), activation='relu')(inputs_b)
    x_b = Dense(128, kernel_initializer=keras.initializers.glorot_uniform(seed=1000), activation='relu')(x_b)
    x_b = Dense(64, kernel_initializer=keras.initializers.glorot_uniform(seed=1000), activation='relu')(x_b)
    outputs_pre = Dense(num_classes, kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs = Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model

# load data
(x_train, y_train), (x_test, y_test) = load_data(args.dataset, False, args.ndata)
num_class=len(y_train[0])
print('-----------------------memGuard------------------------------')

defense_model  = create_defense_classifier(input_shape=(num_class,), num_classes=num_class)

# load target model
target_model_path = f'../models/target/{args.dataset}_{args.ndata}_{args.model}.tf'
target_model = tf.keras.models.load_model(target_model_path)


# test attack model
x_test_attack = np.concatenate([target_model.predict(x_train), target_model.predict(x_test)],
                                 axis=0)
member_test = np.concatenate([np.ones(len(x_train)), np.zeros(len(x_test))], axis=0)

defense_model.fit(x_test_attack, member_test,
                  shuffle=True, batch_size=args.batch_size, epochs=args.epoch, verbose=1)















