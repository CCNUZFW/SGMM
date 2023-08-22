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
from keras.models import load_model
from keras.models import Model


start = time.time()
input_feature = keras.layers.Input(shape=[80, 16, 12, 1])

if __name__ == '__main__':
    # m = build_model()
    m = load_model('Bottle2D.h5')
    m.summary()
    # m.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 数据准备
    # y_train = np.zeros((45*514, 45), 'float')
    # y_test = np.zeros((45*128, 45), 'float')
    #
    # for i in range(45):
    #     y_train[i * 514:(i + 1) * 514, i] = 1
    #     y_test[i * 128:(i + 1) * 128, i] = 1

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

    # X_train = x_train_input.reshape(x_train_input.shape[0], 16, 12, 80, 1)
    # X_train = X_train.transpose(0, 3, 1, 2, 4)
    # # X_train = X_train.reshape(X_train.shape[0] * 80, 16, 12, 1)
    # X_test = x_test_input.reshape(x_test_input.shape[0], 16, 12, 80, 1)
    # X_test = X_test.transpose(0, 3, 1, 2, 4)
    # # X_test = X_test.reshape(X_test.shape[0] * 80, 16, 12, 1)

    X_train = x_train_input.reshape(x_train_input.shape[0], 192, 80)
    X_train = X_train.reshape(X_train.shape[0] * 80, 16, 12, 1)
    X_test = x_test_input.reshape(x_test_input.shape[0], 192, 80)
    X_test = X_test.reshape(X_test.shape[0] * 80, 16, 12, 1)


    # m.fit([X_train], [y_train], epochs=50, batch_size=256)
    # m.save('Bottle.h5')
    # pd.options.display.max_rows = 1000000
    # pd.set_option('display.width', 10000, 'display.max_rows', 10000)

    layer_model = Model(inputs=m.input, outputs=m.get_layer('flatten_1').output)

    train_output = layer_model.predict(X_train)
    doc1 = open('BottleTrain2D.csv', 'w')
    train_write = csv.writer(doc1)
    for i in train_output:
        print(i.shape)
        train_write.writerow(i)
        # print(i, file=doc1)
    doc1.close()

    test_output = layer_model.predict(X_test)
    doc2 = open('BottleTest2D.csv', 'w')
    test_write = csv.writer(doc2)
    for i in test_output:
        print(i.shape)
        test_write.writerow(i)
        # print(i, file=doc2)
    doc2.close()
    # accuracy = m.evaluate([X_test], [y_test])
    # print(m.metrics_names)
    # print('accuracy:', accuracy)

end = time.time()
print("Running time: %d minutes %d seconds " % (((end-start)//60), ((end-start) % 60)))

# out = open('CNNtrain.csv','w')      #结果输入到CSV文件里
# csv_write = csv.writer(out)
# # tout=sess.run(h_fc1_drop,feed_dict={x:X_test ,y_actual: Y_test, keep_prob: 1.0})
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

