import keras
from keras.layers import LSTM, Dense, Dropout, Flatten,Conv2D,Bidirectional,GRU
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import csv


start = time.time()
input_feature = keras.layers.Input(shape=[16, 12, 1])


def scheduler(epoch):
    # 每隔30个epoch，学习率减小为原来的1/10
    if epoch % 30 == 0 and epoch != 0:
        lr = K.get_value(m.optimizer.lr)
        K.set_value(m.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(m.optimizer.lr)


def build_model():
    # # #Stage1：Build CNN-Model
    hidden20 = keras.layers.Conv2D(16, (1, 1), activation='relu', padding='valid', data_format='channels_last',name='layer_con0')(input_feature)
    hidden21 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='valid', data_format='channels_last',name='layer_con1')(hidden20)
    hidden211 = keras.layers.BatchNormalization()(hidden21)
    hidden22 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(hidden211)
    # hidden23 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='valid', data_format='channels_last',name='layer_con2')(hidden22)
    # hidden231 = keras.layers.BatchNormalization()(hidden23)
    # hidden24 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(hidden231)
    hidden25 = keras.layers.Conv2D(16, (1, 1), activation='relu', padding='valid', data_format='channels_last',name='layer_con3')(hidden22)
    hidden251 = keras.layers.BatchNormalization()(hidden25)
    hidden26 = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(hidden251)
    #
    # hidden100 = keras.layers.Reshape((1, 96))(hidden26)
    # hidden101 = keras.layers.Permute((2, 1))(hidden100)

    hidden01 = keras.layers.Flatten()(hidden26)
    hidden04 = keras.layers.Dense(45, activation='softmax', name='layer_softmax')(hidden01)

    # # # # #Stage2：Build RNN-Model
    # hidden51 = Bidirectional(LSTM(128, name='layer_lstm1', return_sequences=True))(input_feature)
    # # hidden52 = Bidirectional(LSTM(192, name='layer_lstm2', return_sequences=True))(hidden51)
    # hidden01 = keras.layers.Flatten()(hidden51)
    # # hidden02 = keras.layers.Dense(1024, activation='relu', name='layer_fully1024')(hidden01)
    # hidden04 = keras.layers.Dense(45, activation='softmax', name='layer_softmax')(hidden01)

    model = keras.models.Model(inputs=[input_feature], outputs=[hidden04])
    return model


if __name__ == '__main__':
    m = build_model()
    m.summary()
    m.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 数据准备
    y_train = np.zeros((45*514*80, 45), 'float')
    y_test = np.zeros((45*128*80, 45), 'float')

    for i in range(45):
        y_train[i * 514*80:(i + 1) * 514*80, i] = 1
        y_test[i * 128*80:(i + 1) * 128*80, i] = 1

    path = 'CLGSV_train.csv'
    inputs_train = pd.read_csv(path, header=None)
    x_train_input = inputs_train.values

    path = 'CLGSV_test.csv'
    inputs_test = pd.read_csv(path, header=None)
    x_test_input = inputs_test.values

    # path = 'BottleTrain.csv'
    # inputs_train = pd.read_csv(path, header=None)
    # x_train_input = inputs_train.values
    #
    # path = 'BottleTest.csv'
    # inputs_test = pd.read_csv(path, header=None)
    # x_test_input = inputs_test.values

    X_train = x_train_input.reshape(x_train_input.shape[0], 192, 80)
    # X_train = X_train.transpose(0, 2, 1)
    # X_train = X_train.transpose(1, 0, 2)
    X_train = X_train.reshape(X_train.shape[0] * 80, 16, 12, 1)
    X_test = x_test_input.reshape(x_test_input.shape[0], 192, 80)
    # X_test = X_test.transpose(0, 2, 1)
    # X_test = X_test.transpose(1, 0, 2)
    X_test = X_test.reshape(X_test.shape[0] * 80, 16, 12, 1)

    reduce_lr = LearningRateScheduler(scheduler)
    m.fit([X_train], [y_train], epochs=50, batch_size=256, callbacks=[reduce_lr], verbose=1)
    m.save('Bottle2D.h5')

    accuracy = m.evaluate([X_test], [y_test])
    print(m.metrics_names)
    print('accuracy:', accuracy)

end = time.time()
print("Running time: %d minutes %d seconds " % (((end-start)//60), ((end-start) % 60)))

# out = open('CNNtrain.csv','w')      #结果输入到CSV文件里
# csv_write = csv.writer(out)
# # tout=sess.run(h_fc1_drop,feed_dict={x:X_test ,y_actual: Y_test, keep_prob: 1.0})
# tout=m.
# lentout=len(tout)
# for i in range(lentout):           #矩阵从0开始的
#     csv_write.writerow(tout[i])
#
# out = open('mfcctrainout.csv','w')      #结果输入到CSV文件里
# csv_write = csv.writer(out)
# tout=sess.run(h_fc1,feed_dict={x:X_train ,y_actual: Y_train, keep_prob: 1.0})
# lentout=len(tout)
# for i in range(lentout):           #矩阵从0开始的
#     csv_write.writerow(tout[i])

