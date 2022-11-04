# based on “https://github.com/hyhmia/BlindMI”


import numpy as np
import tensorflow as tf
import csv


def load_data(dataset,shadow, ndata):
    (x_train, y_train), (x_test, y_test) = globals()['load_' + dataset](shadow,ndata)
    return (x_train, y_train), (x_test, y_test)

def preprocessingCIFAR(toTrainData, toTestData):
    def reshape_for_save(raw_data):
        # raw_data = np.dstack((raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
        # raw_data = raw_data.reshape((raw_data.shape[0], 32, 32, 3)).transpose(0,3,1,2)

        # raw_data = raw_data.transpose(0, 3, 1, 2)
        return raw_data.astype(np.float32)

    offset = np.mean(reshape_for_save(toTrainData), 0)
    scale  = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

    def rescale(raw_data):
        return (reshape_for_save(raw_data) - offset) / scale
    return rescale(toTrainData), rescale(toTestData)

def load_cifar100(shadow,ndata):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    if shadow == False:
        (x_train, y_train), (x_test, y_test) = (x_train[0:ndata], y_train[0:ndata]), \
                                               (x_train[ndata:ndata*2], y_train[ndata:ndata*2])
    elif shadow == True:
        (x_train, y_train), (x_test, y_test) = (x_train[ndata:ndata*2], y_train[ndata:ndata*2]), \
                                               (x_train[:ndata], y_train[:ndata])

    # x_train, x_test = x_train/255.0, x_test/255.0
    x_train, x_test = preprocessingCIFAR(x_train,x_test)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    return (x_train, y_train), (x_test, y_test)


def load_cifar10(shadow,ndata):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if shadow == False:
        (x_train, y_train), (x_test, y_test) = (x_train[0:ndata], y_train[0:ndata]), \
                                               (x_train[ndata:ndata*2], y_train[ndata:ndata*2])
    elif shadow == True:
        (x_train, y_train), (x_test, y_test) = (x_train[ndata:ndata*2], y_train[ndata:ndata*2]), \
                                               (x_train[:ndata], y_train[:ndata])

    # x_train, x_test = x_train / 255.0, x_test / 255.0
    (x_train, x_test) = preprocessingCIFAR(x_train, x_test)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    return (x_train, y_train), (x_test, y_test)

def load_Purchase(shadow, ndata):
    x=[]
    y=[]
    with open('data/purchase', 'r') as infile:
        reader = csv.reader(infile)
        for line in reader:
            y.append(int(line[0]))
            x.append([int(x) for x in line[1:]])
        x = np.array(x)
        y = (np.array(y) - 1).reshape((-1, 1))
        indices = np.arange(len(y))
        # p = np.random.permutation(indices)
        p = indices
        if shadow == False:
            x_train, y_train = x[p[:ndata]], y[p[:ndata]]
            x_test, y_test = x[p[ndata:ndata * 2]], y[p[ndata:ndata * 2]]
        else:
            x_test, y_test = x[p[:ndata]], y[p[:ndata]]
            x_train, y_train = x[p[ndata:ndata * 2]], y[p[ndata:ndata * 2]]

        input_dim = 600
        n_classes = 100
    return (x_train, y_train), (x_test, y_test)

def load_Location(shadow, ndata):
    x=[]
    y=[]
    with open('data/location', 'r') as infile:
        reader = csv.reader(infile)
        for line in reader:
            y.append(int(line[0]))
            x.append([int(x) for x in line[1:]])
        x = np.array(x)
        y = (np.array(y) - 1).reshape((-1, 1))
        indices = np.arange(len(y))
        p = indices
        # p = np.random.permutation(indices)

        if shadow == False:
            x_train, y_train = x[p[:1600]], y[p[:1600]]
            x_test, y_test = x[p[1600:3200]], y[p[1600:3200]]
        else:
            x_test, y_test = x[p[:1600]], y[p[:1600]]
            x_train, y_train = x[p[1600:3200]], y[p[1600:3200]]

        input_dim = 446
        n_classes = 30

    return (x_train, y_train), (x_test, y_test)


def load_Texas(shadow, ndata):
    x = []
    y = []
    with open('data/texas/100/feats', 'r') as infile:
        reader = csv.reader(infile)
        for line in reader:
            x.append([int(x) for x in line[1:]])
            x = np.array(x)
    with open('data/texas/100/labels', 'r') as infile:
        reader = csv.reader(infile)
        for line in reader:
            y.append(int(line[0]))
        y = (np.array(y) - 1).reshape((-1, 1))
        indices = np.arange(len(y))
        p = indices
        # p = np.random.permutation(indices)
        if shadow == False:
            x_train, y_train = x[p[:10000]], y[p[:10000]]
            x_test, y_test = x[p[10000:20000]], y[p[10000:20000]]
        else:
            x_test, y_test = x[p[:10000]], y[p[:10000]]
            x_train, y_train = x[p[10000:20000]], y[p[10000:20000]]
        input_dim = 6168
        n_classes = 100
    return (x_train, y_train), (x_test, y_test)




