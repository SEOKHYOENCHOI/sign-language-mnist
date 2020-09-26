import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
num_classes = 25
image_rows, image_columns = 28, 28
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (300, 200)
# fontScale
fontScale = 3
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2
def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    x = raw[:, 1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, image_rows, image_columns, 1)
    out_x = out_x / 255
    return out_x, out_y
def train():
    train_data = "./sign-language-mnist/sign_mnist_train.csv"
    train_data = np.loadtxt(train_data, delimiter=',', skiprows=1, dtype=float)
    x, y = prep_data(train_data)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(image_rows, image_columns, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(96, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(96, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(x=x, y=y, epochs=50, batch_size=100, validation_split=0.1)
    model.history.history['val_acc'] = model.history.history['val_accuracy']
    model.save('./')
def train2():
    train_data = "./sign-language-mnist/sign_mnist_train.csv"
    train_data = np.loadtxt(train_data, delimiter=',', skiprows=1, dtype=float)
    x, y = prep_data(train_data)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(image_rows, image_columns, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(x=x, y=y, epochs=20, batch_size=100, validation_split=0.1)
    model.history.history['val_acc'] = model.history.history['val_accuracy']
    model.save('./')
def train3():
    train_data = "./sign-language-mnist/sign_mnist_train.csv"
    train_data = np.loadtxt(train_data, delimiter=',', skiprows=1, dtype=float)
    x, y = prep_data(train_data)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(image_rows, image_columns, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), padding='same', strides=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same', strides=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(x=x, y=y, epochs=20, batch_size=100, validation_split=0.1)
    model.history.history['val_acc'] = model.history.history['val_accuracy']
    model.save('./')
def train4():
    train_data = "./sign-language-mnist/sign_mnist_train.csv"
    train_data = np.loadtxt(train_data, delimiter=',', skiprows=1, dtype=float)
    x, y = prep_data(train_data)
    test_data = "./sign-language-mnist/sign_mnist_test.csv"
    test_data = np.loadtxt(test_data, delimiter=',', skiprows=1, dtype=float)
    test_x, test_y = prep_data(test_data)
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(x)
    model = Sequential()
    model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5,
                                                min_lr=0.00001)
    model.fit(datagen.flow(x, y, batch_size=128), epochs=20, validation_data=(test_x, test_y),
              callbacks=[learning_rate_reduction])
    model.history.history['val_acc'] = model.history.history['val_accuracy']
    model.save('./')
def predict():
    model = load_model('./')
    test_data = "./sign-language-mnist/sign_mnist_test.csv"
    test_data = np.loadtxt(test_data, delimiter=',', skiprows=1, dtype=float)
    test_data_x = test_data[:, 1:]
    num_images = test_data.shape[0]
    out_x = test_data_x.reshape(num_images, image_rows, image_columns, 1)
    submission_file_name = './submission.csv'
    if os.path.isfile(submission_file_name):
        os.remove(submission_file_name)
    pred = model.predict(out_x)
    with open(submission_file_name, 'ab') as csv_file:
        np.savetxt(csv_file, [['label']], '%s', delimiter=',')
        pred = np.argmax(pred, axis=-1)
        np.savetxt(csv_file, pred, fmt=['%d'], delimiter=',')
def csv_to_image():
    csv_file_name = './sign-language-mnist/sign_mnist_test.csv'
    csv_file = np.loadtxt(csv_file_name, delimiter=',', skiprows=1, dtype=float)
    images = csv_file[:, 1:]
    for index, image in enumerate(images):
        if index > 10:
            break
        cv2.imwrite('./image/{}.jpg'.format(index), image.reshape(image_rows, image_columns, 1))
def evaluate():
    test_data = "./sign-language-mnist/sign_mnist_test.csv"
    test_data = np.loadtxt(test_data, delimiter=',', skiprows=1, dtype=float)
    test_y = test_data[:, 0]
    predicted_y = "./submission.csv"
    predicted_y = np.loadtxt(predicted_y, delimiter=',', skiprows=1, dtype=float)
    equals_count = 0
    for i in range(len(test_y)):
        test = test_y[i]
        predicted = predicted_y[i]
        if test == predicted:
            equals_count += 1
    accuracy = equals_count / len(test_y) * 100
    print('accuracy: ' + str(accuracy) + '%')
def test():
    model = load_model('saved_model.h5')
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))
    while True:
        retval, image = cap.read()
        if retval:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, dsize=(image_rows, image_columns), interpolation=cv2.INTER_AREA)
            reshaped = resized.reshape(1, image_rows, image_columns, 1)
            res = str(chr(int(model.predict_classes(reshaped)) + ord('A')))
            cv2.putText(image, res, org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('video-color', image)
            cv2.imshow('video-gray', resized)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        else:
            print('error')
    cap.release()
    cv2.destroyAllWindows()
# train4()
# predict()
# evaluate()
test()