
# coding: utf-8 读取特征，载入训练好的网络，测试

import tensorflow as tf
from os import listdir
from os.path import isfile, join
from PIL import Image
from sklearn import preprocessing
import numpy as np
import argparse
import os
from CNN import createModel
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from createTrainingData import createTrainingData

#cpu gpu控制
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# config=tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess=tf.compat.v1.Session(config=config)

#常数
WIDTH = 36#108
HEIGHT = 56#168

depth = 1
num_classes = 2

positiveTestImagePath = './testDataPositive'
negativeTestImagePath = './testDataNegative'


def main(args):
    weights_file = (args.weightsFile)
    positiveImagePath = (args.positiveTestImages)
    negativeImagePath = (args.negativeTestImages)

    createTrainingData(positiveImagePath, negativeImagePath, positiveTestImagePath, negativeTestImagePath)

    #读取特征
    X_V, X_ML, X_HH, X_HS, X_index, Y, imageCount, fileNames = readDatatz(positiveImagePath, negativeImagePath, positiveTestImagePath, negativeTestImagePath)

    X_V = X_V.reshape((imageCount, HEIGHT, WIDTH, depth))
    X_ML = X_ML.reshape((imageCount, HEIGHT, WIDTH, depth))
    X_HH = X_HH.reshape((imageCount, HEIGHT, WIDTH, depth))
    X_HS = X_HS.reshape((imageCount, HEIGHT, WIDTH, depth))

    #载入模型
    CNN_model = createModel(HEIGHT, WIDTH, depth, num_classes)
    CNN_model.load_weights(weights_file)

    #预测
    evaluate(CNN_model,X_V,X_ML,X_HH,X_HS, Y,fileNames,positiveImagePath, negativeImagePath)

def scaleData(inp, minimum, maximum):
    minMaxScaler = preprocessing.MinMaxScaler(copy=True, feature_range=(minimum,maximum))
    inp = inp.reshape(-1, 1)
    inp = minMaxScaler.fit_transform(inp)

    return inp

def readAndScaleImage(f, customStr, trainImagePath, X_V, X_ML, X_HH, X_HS, X_index, Y, sampleIndex, sampleVal):
    fileName = (os.path.splitext(f)[0])

    fV = (f.replace(fileName, fileName + customStr + '_V')).replace('.jpg', '.tiff').replace('.png', '.tiff')
    fML = (f.replace(fileName, fileName + customStr + '_ML')).replace('.jpg', '.tiff').replace('.png', '.tiff')
    fHH = (f.replace(fileName, fileName + customStr + '_HH')).replace('.jpg', '.tiff').replace('.png', '.tiff')
    fHS = (f.replace(fileName, fileName + customStr + '_HS')).replace('.jpg', '.tiff').replace('.png', '.tiff')

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

    imgVector = imgV.reshape(1, WIDTH*HEIGHT)
    X_V[sampleIndex, :] = imgVector
    imgVector = imgML.reshape(1, WIDTH*HEIGHT)
    X_ML[sampleIndex, :] = imgVector
    imgVector = imgHH.reshape(1, WIDTH*HEIGHT)
    X_HH[sampleIndex, :] = imgVector
    imgVector = imgHS.reshape(1, WIDTH*HEIGHT)
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

    positiveCount = len(positiveImageFiles)
    negativeCount = len(negativeImageFiles)

    print('positive samples: ' + str(positiveCount))
    print('negative samples: ' + str(negativeCount))
    imageCount = positiveCount + negativeCount
    #intialization
    X_V = np.zeros((positiveCount + negativeCount, WIDTH*HEIGHT))
    X_ML = np.zeros((positiveCount + negativeCount, WIDTH*HEIGHT))
    X_HH = np.zeros((positiveCount + negativeCount, WIDTH*HEIGHT))
    X_HS = np.zeros((positiveCount + negativeCount, WIDTH*HEIGHT))
    X_index = np.zeros((positiveCount + negativeCount, 1))
    Y = np.zeros((positiveCount + negativeCount, 1))
    
    sampleIndex = 0

    sampleIndex = readImageSet(positiveImageFiles, positiveTrainImagePath, X_V, X_ML, X_HH, X_HS, X_index, Y, sampleIndex, 0)
    print('positive data loaded.')

    sampleIndex = readImageSet(negativeImageFiles, negativeTrainImagePath, X_V, X_ML, X_HH, X_HS, X_index, Y, sampleIndex, 1)
    print('negative data loaded.')

    print('Total Samples Loaded: ', sampleIndex)

    fileNames = positiveImageFiles + negativeImageFiles 

    return X_V, X_ML, X_HH, X_HS, X_index, Y, imageCount, fileNames

def evaluate(model, X_V_test,X_ML_test,X_HH_test,X_HS_test,y_test,fileNames,positiveImagePath, negativeImagePath):

    model_out = model.predict([X_V_test,X_ML_test,X_HH_test,X_HS_test])
    passCnt = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    directory = './moire pattern'

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # 如果是文件，删除文件
        if os.path.isfile(file_path):
            os.remove(file_path)

    for i in range(len(y_test)):

        print(model_out[i, :])
        print(y_test[i])
        print(',')

        if np.argmax(model_out[i, :]) == y_test[i]:
            str_label='Pass'
            passCnt = passCnt + 1
        else:
            str_label='Fail'

        if y_test[i] ==0:
            if np.argmax(model_out[i, :]) == y_test[i]:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if np.argmax(model_out[i, :]) == y_test[i]:
                TN = TN + 1
            else:
                FP = FP + 1

        if np.argmax(model_out[i, :]) == 0:

            # makwen(fileNames[i],positiveImagePath, negativeImagePath)
            try:
                image = Image.open(join(positiveImagePath, fileNames[i]))

            except Exception:
                image = Image.open(join(negativeImagePath, fileNames[i]))

            image.save(directory +'/'+ fileNames[i])

             

    start = "\033[1m"
    end = "\033[0;0m"
    print(start + 'confusion matrix (test / validation)' + end)
    print(start + 'true positive:  '+ end + str(TP))
    print(start + 'false positive: '+ end + str(FP))
    print(start + 'true negative:  '+ end + str(TN))
    print(start + 'false negative: '+ end + str(FN))
    print('\n')
    print(start + 'accuracy:  ' + end + "{:.4f} %".format(100*(TP+TN)/(TP+FP+FN+TN)))
    # print(start + 'precision: ' + end + "{:.4f} %".format(100*TP/(TP + FP)))
    # print(start + 'recall:  ' + end + "{:.4f} %".format(100*TP/(TP + FN)))
    # print('\n')
    print('predict label:'+ end + str(np.around(model_out[:,0])))

# def makwen(name,positiveImagePath, negativeImagePath):
#
#     try:
#         image = Image.open(join(positiveImagePath, name))
#
#     except Exception:
#         image = Image.open(join(negativeImagePath, name))
#
#     img=tohigh(toimage_max_labol(image))
#     img = np.array(img)
#     h, w = img.shape
#     img = img.astype(float)
#
#     wenh = h//4
#     wenw = w//4
#
#     mh = 0
#     mw = 0
#     mm = 1
#     for x in range(0, h-wenh,20):
#         for y in range(0, w-wenw,20):
#             mk = np.sum(img[x:x+wenh-1,y:y+wenw-1])
#             if mm <= mk:
#                 mm = mk
#                 mh = x
#                 mw = y
#
#
#     bbox = [mw, mh,wenw,wenh]
#
#     box = Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2], height=bbox[3], fill=False, edgecolor='red', linewidth=2)
#
#     fig = plt.imshow(image)
#     plt.title(name), plt.xticks([]), plt.yticks([])
#
#     fig.axes.add_patch(box)
#
#     fig.axes.text(box.xy[0],box.xy[1],'moire',va='center', ha='center', fontsize=16, color='white',
#                        bbox=dict(facecolor='red', lw=0))
#     plt.savefig('./moire pattern/'+name)
#     # plt.show()
#     plt.clf()  # 添加上这一行，画完第一个图后，将plt重置


def tohigh(img):
    bl = 1/5
    f = np.fft.fft2(img)#傅里叶变换
    fshift = np.fft.fftshift(f)
    #设置高通滤波器
    rows, cols = img.shape
    fshift[int(rows*bl):int(rows*(2-bl)), int(rows*bl):int(cols*(2-bl))] = 0
    #傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    return iimg

def toimage_max_labol(img):
    img = np.array(img)
    h, w, q = img.shape
    img = img.astype(float)
    image_max = np.max(img, axis=2)
    image_max_labol = np.zeros([h, w])
    for x in range(0, h):
        for y in range(0, w):
            for z in range(0, q):
                if img[x,y,z] == image_max[x,y]:
                    image_max_labol[x,y] = z / 2      
    return image_max_labol

    
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('weightsFile', type=str, help='saved CNN model file')

    parser.add_argument('positiveTestImages', type=str, help='Directory with positive (Moiré pattern) images.')
    parser.add_argument('negativeTestImages', type=str, help='Directory with negative (Normal) images.')


    return parser.parse_args(argv)


if __name__ == '__main__':

    main(parse_arguments(['./moirePatternCNN.h5',         #模型h5文件
                          './data/newm',           #测试摩尔纹图片目录
                          './data/empty']))                   #测试正常图片目录

