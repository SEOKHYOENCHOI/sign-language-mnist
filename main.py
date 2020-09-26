import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
# image augmentation 하는 툴
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau

import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (300, 200)
# fontScale
fontScale = 3
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2
import numpy as np

# Load ASL Dataset
# label과 이미지 pixel값의 나열임

num_classes = 25
image_rows, image_columns = 28, 28

def train():
    train_df = pd.read_csv("input/sign_mnist_train/sign_mnist_train.csv")
    test_df = pd.read_csv("input/sign_mnist_test/sign_mnist_test.csv")

    test = pd.read_csv("input/sign_mnist_test/sign_mnist_test.csv")
    y = test['label']

    # label 값은 y 값으로 빼놓고 dataset에서 삭제함
    y_train = train_df['label']
    y_test = test_df['label']
    del train_df['label']
    del test_df['label']

    #  텍스트 범주에서 숫자형 범주로, 숫자형 범주에서 원핫인코딩으로
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    x_train = train_df.values
    x_test = test_df.values

    # 빠른 데이터 처리를 위해 이진화 시킴
    x_train = x_train / 255
    x_test = x_test / 255

    # Reshaping the data from 1-D to 3-D as required through input by CNN's
    # 28*28*1 꼴의 데이터로 바뀜
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    f, ax = plt.subplots(2,5)
    f.set_size_inches(10, 10)
    k = 0
    for i in range(2):
        for j in range(5):
            ax[i,j].imshow(x_train[k].reshape(28, 28) , cmap = "gray")
            k += 1
        plt.tight_layout()


    # With data augmentation to prevent overfitting

    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images


    datagen.fit(x_train)

    # Train Model

    # 이 콜백을 사용하면 검증 손실이 향상되지 않을 때 학습률을 작게 할 수 있습니다
    # factor = factor by which the learning rate will be reduced. new_lr = lr * factor.
    # patience = 얼마만큼 epoch을 지켜볼것인지
    # verbose = 메시지를 생성할 것인지 말 것인지
    # min_lr = 최소 learning rate
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

    model = Sequential()
    model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 512 , activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 24 , activation = 'softmax'))
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    model.summary()

    history = model.fit(datagen.flow(x_train,y_train, batch_size = 64) ,epochs = 30, validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])

    print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

    model.save('saved_model.h5')

    # J랑 Z가 빠지므로
    predictions = model.predict_classes(x_test)
    print(type(predictions))
    for i in range(len(predictions)):
        if(predictions[i] >= 9):
            predictions[i] += 1


    df = pd.DataFrame(data={"letter": predictions})
    df.to_csv("./submission.csv", sep=',',index=False)

    classes = ["Class " + str(i) for i in range(25) if i != 9]
    print(classification_report(y, predictions, target_names = classes))

def test():
    model = load_model('saved_model.h5')
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))
    while True:
        retval, image = cap.read()
        if retval:
            image = image[:, 0:480]
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

# train()
test()
