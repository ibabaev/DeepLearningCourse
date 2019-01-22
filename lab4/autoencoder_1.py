import glob
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from matplotlib import pyplot as plt
import h5py
import numpy as np
import pandas as pd
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical, np_utils
from skimage import io, color, exposure, transform
from sklearn.model_selection import train_test_split
K.set_image_data_format('channels_last')

NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    img = color.rgb2gray(img)
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])



def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(1, IMG_SIZE, IMG_SIZE)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model

def encoder(input_img):
    #encoder
    #input = 48 x 48 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #48 x 48 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #24 x 24 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #24 x 24 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #12 x 12 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #12 x 12 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #12 x 12 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #12 x 12 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #12 x 12 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #24 x 24 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 24 x 24 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 48 x 48 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(NUM_CLASSES, activation='softmax')(den)
    return out

if __name__ == '__main__':

    try:
        with  h5py.File('X.h5') as hf:
            X, Y = hf['imgs'][:], hf['labels'][:]

        print("Loaded images from X.h5")

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

        with h5py.File('X.h5', 'w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y)


    lr = 0.01

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

    #X_train = X.reshape(X.shape[0], 1, IMG_SIZE, IMG_SIZE)
    #X_test = X_test.reshape(X_test.shape[0], 1, IMG_SIZE, IMG_SIZE)
    X_train = np.reshape(X, [-1, IMG_SIZE, IMG_SIZE, 1])
    X_test = np.reshape(X_test, [-1, IMG_SIZE, IMG_SIZE, 1])

    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)

    Y_train = np_utils.to_categorical(Y, NUM_CLASSES)
    Y_test = np_utils.to_categorical(Y_test, NUM_CLASSES)

    print(np.max(X_train), np.max(X_test))
#################DATA READY##########################################################################

    batch_size = 64
    epochs = 30

    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    num_classes = NUM_CLASSES

    autoencoder = Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

    autoencoder.summary()

    #prepareData for AUTOENCODER

    train_X, valid_X, train_ground, valid_ground = train_test_split(X_train,
                                                                    X_train,
                                                                    test_size=0.2,
                                                                    random_state=13)


    autoencoder_train = autoencoder.fit(train_X, train_ground,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=(valid_X, valid_ground))

    print("here")

    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(20)

    autoencoder.save_weights('autoencoder.h5')

    # Change the labels from categorical to one-hot encoding
    train_Y_one_hot = Y_train
    test_Y_one_hot = Y_test

    train_X, valid_X, train_label, valid_label = train_test_split(X_train, train_Y_one_hot,
                                                                  test_size=0.2,
                                                                  random_state=13)

    print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

    encode = encoder(input_img)
    full_model = Model(input_img, fc(encode))

    for l1, l2 in zip(full_model.layers[:19], autoencoder.layers[0:19]):
        l1.set_weights(l2.get_weights())

    print(autoencoder.get_weights()[0][1])

    print(full_model.get_weights()[0][1])

    for layer in full_model.layers[0:19]:
        layer.trainable = False

    full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    full_model.summary()

    classify_train = full_model.fit(X_train, Y_train,
              epochs=20,
              batch_size=128,
              shuffle=True,
              verbose=1,
              validation_data=(X_test, Y_test))

    full_model.save_weights('autoencoder_classification.h5')

    for layer in full_model.layers[0:19]:
        layer.trainable = True

    full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    classify_train = full_model.fit(X_train, Y_train,
              epochs=20,
              batch_size=128,
              shuffle=True,
              verbose=1,
              validation_data=(X_test, Y_test))

    full_model.save_weights('classification_complete.h5')


    accuracy = classify_train.history['acc']
    val_accuracy = classify_train.history['val_acc']
    loss = classify_train.history['loss']
    val_loss = classify_train.history['val_loss']
    epochs = range(len(accuracy))



    score = full_model.evaluate(X_test, Y_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

#last step:  loss: 0.0092 - acc: 0.9981 - val_loss: 0.8978 - val_acc: 0.9170
#Test loss: 0.897757806686944
#Test accuracy: 0.9170229612506767