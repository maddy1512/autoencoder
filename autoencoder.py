from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import os, pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.manifold import TSNE
import glob

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from PIL import Image
from keras.applications import imagenet_utils
from keras.callbacks import TensorBoard
from keras.models import load_model



tf.app.flags.DEFINE_string('image_path', '/Users/maitraythaker/projects/image_similarity/images_downloaded',
                           'Addres of all images')
tf.app.flags.DEFINE_integer('no_of_images', 987, 'Maximum number of images')
tf.app.flags.DEFINE_boolean('stretched', False,
                            'Determines if the resulting merged image to be stretched')
tf.app.flags.DEFINE_integer('image_width', 128,
                            'width and height of each image in the resulting merged image')


def main(_):
    # get all images and convert them to 100x100x1 for test and train 
    image_names = glob.glob(tf.flags.FLAGS.image_path + "/*.jpg")

    _data_train = np.zeros([len(image_names[1:101]), 100,100,1])
    _test = np.zeros([len(image_names[102:112]), 100,100,1])


    for i, name in enumerate(image_names[1:101]):
        dd = Image.open(name)
        _data_train[i, :] = prepare_image(dd,target=(100,100))
    print _data_train.shape
    for i,name in enumerate(image_names[102:112]):
        dd = Image.open(name)
        _test[i, :] = prepare_image(dd, target=(100, 100))
    # _data[]
    train(_data_train,_test)



        # image = Image.open()
    # image = prepare_image(image, target=(224, 224))



def train(train,test_data):
    if os.path.exists("auto_encoder.h5"):
        autoencoder = load_model("auto_encoder.h5")
    else:
        input_img = Input(shape=(100, 100, 1))  # adapt this if using `channels_first` image data format

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        # here we ask our target label is train data itself so what this network does is that it learns how to reconstruct its input
        # TODO add sparse encoding regularization and test the difference
        autoencoder.fit(train, train,
                        epochs=100,
                        batch_size=10,
                        shuffle=True,
                        validation_split=0.1,
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        autoencoder.save("auto_encoder.h5")

    decoded_imgs = autoencoder.predict(test_data)
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1,n):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(test_data[i].reshape(100, 100))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(100, 100))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def prepare_image(image, target):
    if image.mode != "L":
        image = image.convert("L")

    # resize the input image
    image = image.resize(target)
    image = img_to_array(image).astype("float32")/255.
    return image

if __name__ == '__main__':
    tf.app.run()