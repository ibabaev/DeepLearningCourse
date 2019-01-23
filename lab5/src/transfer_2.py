import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from skimage import io, color, exposure, transform
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K, applications, Model

K.set_image_data_format('channels_last')

from matplotlib import pyplot as plt

NUM_CLASSES = 43
IMG_SIZE = 48


def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
          centre[1] - min_side // 2:centre[1] + min_side // 2,
          :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])


if __name__ == '__main__':
    try:
        with  h5py.File('X3.h5') as hf:
            X, Y = hf['imgs'][:], hf['labels'][:]

        print("Loaded images from X3.h5")

    except (IOError, OSError, KeyError):
        print("Error in reading X.h5. Processing all images...")
        root_dir = 'GTSRB\Final_Training\Images\\'
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            try:
                img = preprocess_img(io.imread(img_path))
                label = get_class(img_path)
                imgs.append(img)
                labels.append(label)

                if len(imgs) % 1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            except (IOError, OSError):
                print('missed', img_path)
                pass

        X = np.array(imgs, dtype='float32')
        Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

        with h5py.File('X3.h5', 'w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y)
    ############################################
    lr = 0.01

    model_start = applications.ResNet50(weights="imagenet", include_top=False, input_shape = (IMG_SIZE, IMG_SIZE, 3))
    model_start.summary()

    #fixed weights
    for layer in model_start.layers:
        layer.trainable = False

    x = model_start.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # creating the final model
    model = Model(input=model_start.input, output=predictions)

    ##################################################
    test = pd.read_csv('GT-final_test.csv', sep=';')
    X_test = []
    y_test = []

    i = 0
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('GTSRB\Final_Test\Images\\', file_name)
        X_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)

    X_test = np.array(X_test)
    Y_test = np.array(y_test)

    print("Training matrix shape", X.shape)
    print("Training matrix class shape", Y.shape)
    print("Testing matrix shape", X_test.shape)

    X_train = np.reshape(X, [-1, IMG_SIZE, IMG_SIZE, 3])
    X_test = np.reshape(X_test, [-1, IMG_SIZE, IMG_SIZE, 3])

    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)

    Y_train = np_utils.to_categorical(Y, NUM_CLASSES)
    Y_test = np_utils.to_categorical(Y_test, NUM_CLASSES)

    print(np.max(X_train), np.max(X_test))
    #################DATA READY##########################################################################

    batch_size = 64
    epochs = 30
    eh = 20

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Let's train
    model.fit(X_train, Y_train,
              epochs=2,
              batch_size=128,
              shuffle=True,
              verbose=1,
              validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.summary()

#