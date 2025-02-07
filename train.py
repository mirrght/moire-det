
# coding: utf-8 读取特征，训练卷积神经网络

from matplotlib import pyplot as plt
import numpy as np
import argparse
from os import listdir
from os.path import isfile, join
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from CNN import createModel
from tensorflow.python.keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os
import time
from createTrainingData import createTrainingData


#cpu gpu控制
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# config=tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess=tf.compat.v1.Session(config=config)

# 常数
WIDTH = 36  # 108
HEIGHT = 56  # 168


def train_main(args):
    positiveImagePath = (args.positiveImages)
    negativeImagePath = (args.negativeImages)
    numEpochs = (args.epochs)
    positiveTrainImagePath = args.trainingDataPositive
    negativeTrainImagePath = args.trainingDataNegative

    createTrainingData(positiveImagePath, negativeImagePath, positiveTrainImagePath, negativeTrainImagePath)

    # 读取特征图片
    X_V, X_ML, X_HH, X_HS, X_index, Y, imageCount = readDatatz(positiveImagePath, negativeImagePath,
                                                               positiveTrainImagePath, negativeTrainImagePath)

    # 切分训练集测试集
    X_V_train, X_ML_train, X_HH_train, X_HS_train, Y_train, X_V_test, X_ML_test, X_HH_test, X_HS_test, Y_test = trainTestSplit(
        X_V, X_ML, X_HH, X_HS, X_index, Y, imageCount)


    # 实例化模型
    model = trainCNNModel(X_V_train, X_ML_train, X_HH_train, X_HS_train, Y_train,
                          X_V_test, X_ML_test, X_HH_test, X_HS_test, Y_test, numEpochs)

    # 估计测试集
    evaluate(model, X_V_test, X_ML_test, X_HH_test, X_HS_test, Y_test)


def scaleData(inp, minimum, maximum):
    minMaxScaler = preprocessing.MinMaxScaler(copy=True, feature_range=(minimum, maximum))
    inp = inp.reshape(-1, 1)
    inp = minMaxScaler.fit_transform(inp)

    return inp


def readAndScaleImage(f, customStr, trainImagePath, X_V, X_ML, X_HH, X_HS, X_index, Y, sampleIndex, sampleVal):
    fileName = (os.path.splitext(f)[0])

    fV = (f.replace(fileName, fileName + customStr + '_V')).replace('.jpg', '.tiff').replace('.png','.tiff')
    fML = (f.replace(fileName, fileName + customStr + '_ML')).replace('.jpg', '.tiff').replace('.png','.tiff')
    fHH = (f.replace(fileName, fileName + customStr + '_HH')).replace('.jpg', '.tiff').replace('.png','.tiff')
    fHS = (f.replace(fileName, fileName + customStr + '_HS')).replace('.jpg', '.tiff').replace('.png','.tiff')

    try:
        imgV = Image.open(join(trainImagePath, fV))
        imgML = Image.open(join(trainImagePath, fML))
        imgHH = Image.open(join(trainImagePath, fHH))
        imgHS = Image.open(join(trainImagePath, fHS))
    except Exception as e:
        print('Error: Couldnt read the file {}. Make sure only images are present in the folder'.format(fileName))
        print('Exception:', e)
        return None

    imgV = np.array(imgV)
    imgML = np.array(imgML)
    imgHH = np.array(imgHH)
    imgHS = np.array(imgHS)
    imgV = scaleData(imgV, 0, 1)
    imgML = scaleData(imgML, -1, 1)
    imgHH = scaleData(imgHH, -1, 1)
    imgHS = scaleData(imgHS, -1, 1)

    imgVector = imgV.reshape(1, WIDTH * HEIGHT)
    X_V[sampleIndex, :] = imgVector
    imgVector = imgML.reshape(1, WIDTH * HEIGHT)
    X_ML[sampleIndex, :] = imgVector
    imgVector = imgHH.reshape(1, WIDTH * HEIGHT)
    X_HH[sampleIndex, :] = imgVector
    imgVector = imgHS.reshape(1, WIDTH * HEIGHT)
    X_HS[sampleIndex, :] = imgVector

    Y[sampleIndex, 0] = sampleVal
    X_index[sampleIndex, 0] = sampleIndex

    return True


def readImageSet(imageFiles, trainImagePath, X_V, X_ML, X_HH, X_HS, X_index, Y, sampleIndex, bClass):
    for f in imageFiles:
        ret = readAndScaleImage(f, '', trainImagePath, X_V, X_ML, X_HH, X_HS, X_index, Y, sampleIndex, bClass)
        if ret == True:
            sampleIndex = sampleIndex + 1


    return sampleIndex


def readDatatz(positiveImagePath, negativeImagePath, positiveTrainImagePath, negativeTrainImagePath):
    # get augmented, balanced training data image files by class
    positiveImageFiles = [f for f in listdir(positiveImagePath) if (isfile(join(positiveImagePath, f)))]
    negativeImageFiles = [f for f in listdir(negativeImagePath) if (isfile(join(negativeImagePath, f)))]

    positiveCount = len(positiveImageFiles) * 1
    negativeCount = len(negativeImageFiles) * 1

    print('positive samples: ' + str(positiveCount))
    print('negative samples: ' + str(negativeCount))
    imageCount = positiveCount + negativeCount
    # intialization
    X_V = np.zeros((positiveCount + negativeCount, WIDTH * HEIGHT), 'float32')
    X_ML = np.zeros((positiveCount + negativeCount, WIDTH * HEIGHT), 'float32')
    X_HH = np.zeros((positiveCount + negativeCount, WIDTH * HEIGHT), 'float32')
    X_HS = np.zeros((positiveCount + negativeCount, WIDTH * HEIGHT), 'float32')
    X_index = np.zeros((positiveCount + negativeCount, 1), 'float32')
    Y = np.zeros((positiveCount + negativeCount, 1), 'float32')

    sampleIndex = 0

    sampleIndex = readImageSet(positiveImageFiles, positiveTrainImagePath, X_V, X_ML, X_HS, X_HS, X_index, Y,
                               sampleIndex, 0)
    print('positive data loaded.')

    sampleIndex = readImageSet(negativeImageFiles, negativeTrainImagePath, X_V, X_ML, X_HS, X_HS, X_index, Y,
                               sampleIndex, 1)
    print('negative data loaded.')

    print('Total Samples Loaded: ', sampleIndex)
    print(X_V)
    print(X_ML)
    print(Y)

    return X_V, X_ML, X_HH, X_HS, X_index, Y, imageCount


# 基于索引分割多输入数据集
def splitTrainTestDataForBands(inputData, X_train_ind, X_test_ind):
    X_train = np.zeros((len(X_train_ind), WIDTH * HEIGHT), 'float32')
    for i in range(len(X_train_ind)):
        X_train[i, :] = inputData[int(X_train_ind[i, 0]), :]

    X_test = np.zeros((len(X_test_ind), WIDTH * HEIGHT), 'float32')
    for i in range(len(X_test_ind)):
        X_test[i, :] = inputData[int(X_test_ind[i, 0]), :]

    return X_train, X_test


def countPositiveSamplesAfterSplit(trainData):
    count = 0
    for i in range(len(trainData)):
        if (trainData[i, 0] == 0):
            count = count + 1
    return count


def trainTestSplit(X_V, X_ML, X_HH, X_HS, X_index, Y, imageCount):
    testCountPercent = 0.25  # 测试集比例

    #    通过分割成训练集和测试集来评估模型
    X_train_ind, X_test_ind, y_train, y_test = train_test_split(X_index, Y, test_size=testCountPercent, random_state=1,
                                                                stratify=Y)
    print(len(X_train_ind))
    X_V_train, X_V_test = splitTrainTestDataForBands(X_V, X_train_ind, X_test_ind)
    X_ML_train, X_ML_test = splitTrainTestDataForBands(X_ML, X_train_ind, X_test_ind)
    X_HH_train, X_HH_test = splitTrainTestDataForBands(X_HH, X_train_ind, X_test_ind)
    X_HS_train, X_HS_test = splitTrainTestDataForBands(X_HS, X_train_ind, X_test_ind)

    imageHeight = HEIGHT
    imageWidth = WIDTH

    print(countPositiveSamplesAfterSplit(y_train))
    print(len(X_V_train))
    print(len(y_train))
    print(len(X_V_test))
    print(len(y_test))

    num_train_samples = len(y_train)
    print('num_train_samples', num_train_samples)
    X_V_train = X_V_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_V_test = X_V_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))
    print(len(X_V_test))

    X_ML_train = X_ML_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_ML_test = X_ML_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))

    X_HH_train = X_HH_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_HH_test = X_HH_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))

    X_HS_train = X_HS_train.reshape((num_train_samples, imageHeight, imageWidth, 1))
    X_HS_test = X_HS_test.reshape((imageCount - num_train_samples, imageHeight, imageWidth, 1))

    num_train, height, width, depth = X_V_train.shape
    num_test = X_V_test.shape[0]
    num_classes = len(np.unique(y_train))
    print(num_classes)
    return X_V_train, X_ML_train, X_HH_train, X_HS_train, y_train, X_V_test, X_ML_test, X_HH_test, X_HS_test, y_test


def trainCNNModel(X_V_train, X_ML_train, X_HH_train, X_HS_train, y_train,
                  X_V_test, X_ML_test, X_HH_test, X_HS_test, y_test, num_epochs):
    batch_size = 32  # 在每次迭代中，考虑32个训练样本
    num_train, height, width, depth = X_V_train.shape
    num_classes = len(np.unique(y_train))
    Y_train = np_utils.to_categorical(y_train, num_classes)  # One-hot encode the labels
    Y_test = np_utils.to_categorical(y_test, num_classes)  # One-hot encode the labels

    checkPointFolder = 'checkPoint'
    checkpoint_name = checkPointFolder + '/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    if not os.path.exists(checkPointFolder):
        os.makedirs(checkPointFolder)

    model = createModel(height, width, depth, num_classes)

    model.compile(loss='categorical_crossentropy',  # cross-entropy loss function
                  optimizer='adam',  # Adam optimiser
                  metrics=['accuracy'])  # 获取accuracy

    score = np.zeros(num_epochs)
    acc = np.zeros(num_epochs)
    for iter_e in range(num_epochs):
        print('epochs:', iter_e)
        model.fit([X_V_train, X_ML_train, X_HH_train, X_HS_train], Y_train,  # 使用训练集训练
                  batch_size=batch_size, epochs=1,
                  verbose=1, validation_split=0.1, callbacks=callbacks_list)  # 留出训练集10%的数据进行验证
        score[iter_e], acc[iter_e] = model.evaluate([X_V_test, X_ML_test, X_HH_test, X_HS_test], Y_test,
                                                    verbose=1)  # 评估测试集

    # 画图，保存h5模型
    plt.figure(figsize=(10, 10), dpi=100)
    net_figure, ax = plt.subplots(2, 1)
    ax[0].plot(score)
    ax[0].set_title(r'loss')
    ax[1].plot(acc)
    ax[1].set_title(r'acc')
    time_tuple = time.localtime(time.time())
    figure_name = "结束时间为{}年{}月{}日{}点{}分{}秒.png".format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3],
                                                       time_tuple[4], time_tuple[5])
    plt.tight_layout()
    net_figure.savefig('./net_figure/' + figure_name, dpi=100, bbox_inches='tight')

    model.save('./net_wight/结束时间为{}年{}月{}日{}点{}分{}秒.h5'.format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3],
                                                       time_tuple[4], time_tuple[5]))

    return model


def evaluate(model, X_V_test, X_ML_test, X_HH_test, X_HS_test, y_test):
    model_out = model.predict([X_V_test, X_ML_test, X_HH_test, X_HS_test])
    passCnt = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_test)):
        if np.argmax(model_out[i, :]) == y_test[i]:
            str_label = 'Pass'
            passCnt = passCnt + 1
        else:
            str_label = 'Fail'

        if y_test[i] == 0:
            if np.argmax(model_out[i, :]) == y_test[i]:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if np.argmax(model_out[i, :]) == y_test[i]:
                TN = TN + 1
            else:
                FP = FP + 1

    start = "\033[1m"
    end = "\033[0;0m"
    print(start + 'confusion matrix (test / validation)' + end)
    print(start + 'true positive:  ' + end + str(TP))
    print(start + 'false positive: ' + end + str(FP))
    print(start + 'true negative:  ' + end + str(TN))
    print(start + 'false negative: ' + end + str(FN))
    print('\n')
    print(start + 'accuracy:  ' + end + "{:.4f} %".format(100 * (TP + TN) / (TP + FP + FN + TN)))
    print(start + 'precision: ' + end + "{:.4f} %".format(100 * TP / (TP + FP)))
    print(start + 'recall:  ' + end + "{:.4f} %".format(100 * TP / (TP + FN)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('positiveImages', type=str, help='Directory with original positive (Moiré pattern) images.')
    parser.add_argument('negativeImages', type=str, help='Directory with original negative (Normal) images.')

    parser.add_argument('trainingDataPositive', type=str,
                        help='Directory with transformed positive (Moiré pattern) images.')
    parser.add_argument('trainingDataNegative', type=str, help='Directory with transformed negative (Normal) images.')

    parser.add_argument('epochs', type=int, help='Number of epochs for training')

    return parser.parse_args(argv)


if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))
    # main(parse_arguments(['./data/摩尔纹视频图片',  #用于训练的摩尔纹图片目录
    #                       './data/10001training',    #用于训练的正常图片目录
    #                       './trainDataPositive',     #用于训练的摩尔纹图片特征图层目录
    #                       './trainDataNegative',     #用于训练的正常图片特征图层目录
    #                       '50']))
    train_main(parse_arguments(['F:/ta/摩尔纹课题/src/data/MoireIDT/positive/screen',  #用于训练的摩尔纹图片目录 VINmoire MoireIDT screen MoireFace MRBI FHDMi
                          'F:/ta/摩尔纹课题/src/data/MoireIDT/negative',    #用于训练的正常图片目录
                          './trainDataPositive',     #用于训练的摩尔纹图片特征图层目录
                          './trainDataNegative',     #用于训练的正常图片特征图层目录
                          '100']))
