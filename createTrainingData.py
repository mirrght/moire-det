
# 特征提取并保存

import argparse
import numpy as np
import os, sys, heapq
import shutil

from os import listdir
from os.path import isfile, join
from PIL import Image
from math import sqrt

# 特征图片存储目录
positiveTrainImagePath = ''
negativeTrainImagePath = ''

# 常数
rows, cols = 36, 56  # 108, 168


def main(args):
    global positiveTrainImagePath
    global negativeTrainImagePath

    positiveImagePath = (args.positiveImages)
    negativeImagePath = (args.negativeImages)

    if (args.train == 0):
        positiveTrainImagePath = './trainDataPositive'
        negativeTrainImagePath = './trainDataNegative'
    else:
        positiveTrainImagePath = './testDataPositive'
        negativeTrainImagePath = './testDataNegative'

    createTrainingData(positiveImagePath, negativeImagePath, positiveTrainImagePath, negativeTrainImagePath)


# 特征提取
def trans(img):
    
    image_max_labol = tohigh(toimage_max_labol(img))
    img_hsv = img.convert('HSV')
    h, s, v = img_hsv.split()
    h = tohigh(h)
    s = tohigh(s)

    image_max_labol = tohigh(toimage_max_labolr(img))
    # s = tohigh(toimage_max_labolg(img))
    # image_max_labol = tohigh(toimage_max_labolb(img))
    # s = np.zeros_like(h)

    return v, image_max_labol, h, s


# 计算最大通道特征
def toimage_max_labol(img):
    img = np.array(img)
    h, w, q = img.shape
    img = img.astype(float)
    image_max = np.max(img, axis=2)
    image_max_labol = np.zeros([h, w])
    for x in range(0, h):
        for y in range(0, w):
            for z in range(0, q):
                if img[x][y][z] == image_max[x][y]:
                    image_max_labol[x][y] = z / 2

    return image_max_labol

def toimage_max_labolr(img):
    img = np.array(img)
    h, w, q = img.shape
    img = img.astype(float)
    image_max = np.max(img, axis=2)
    image_max_labol = np.zeros([h, w])
    for x in range(0, h):
        for y in range(0, w):
            if img[x,y,0] == image_max[x,y]:
                    image_max_labol[x,y] = 1

    return image_max_labol

def toimage_max_labolg(img):
    img = np.array(img)
    h, w, q = img.shape
    img = img.astype(float)
    image_max = np.max(img, axis=2)
    image_max_labol = np.zeros([h, w])
    for x in range(0, h):
        for y in range(0, w):
            if img[x,y,1] == image_max[x,y]:
                    image_max_labol[x,y] = 1

    return image_max_labol

def toimage_max_labolb(img):
    img = np.array(img)
    h, w, q = img.shape
    img = img.astype(float)
    image_max = np.max(img, axis=2)
    image_max_labol = np.zeros([h, w])
    for x in range(0, h):
        for y in range(0, w):
            if img[x,y,2] == image_max[x,y]:
                    image_max_labol[x,y] = 1

    return image_max_labol


def towen(img):
    img = np.array(img)
    h, w = img.shape
    img = img.astype(float)

    wenh = h // 3
    wenw = w // 3

    mh = 0
    mw = 0
    mm = 1
    for x in range(0, h - wenh):
        for y in range(0, w - wenw):
            mk = np.sum(img[x:x + wenh - 1, y:y + wenw - 1])
            if mm <= mk:
                mm = mk
                mh = x
                mw = y

    return img[mh:mh + wenh, mw:mw + wenw]

# 高通滤波
# def tohigh(img):
#
#     img = np.array(img)
#     bl = 1/5
#     f = np.fft.fft2(img)#傅里叶变换
#     fshift = np.fft.fftshift(f)
#     #设置高通滤波器
#     rows, cols = img.shape
# #     crow,ccol = int(rows*bl), int(cols*bl)
#     fshift[int(rows*bl):int(rows*(2-bl)), int(rows*bl):int(cols*(2-bl))] = 0
#     #傅里叶逆变换
#     ishift = np.fft.ifftshift(fshift)
#     iimg = np.fft.ifft2(ishift)
#     iimg = np.abs(iimg)
#     return iimg

def tohigh(img):

    img = np.array(img)
    f = np.fft.fft2(img)  # 快速傅里叶变换算法得到频率分布
    fshift = np.fft.fftshift(f)  # 将原点转移到中间位置
    transfor_matrix = np.zeros(img.shape)

    d = 5

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
    for i in range(rows):
        for j in range(cols):
            dis = sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            transfor_matrix[i, j] = 1 - np.exp(-dis ** 2 / (2 * d ** 2))

    f1_shift = fshift * transfor_matrix
    f_ishift = np.fft.ifftshift(f1_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


def transformImageAndSave(image, f, customStr, path):
    cA, cH, cV, cD = trans(image)

    fileName = (os.path.splitext(f)[0])

    fV = (f.replace(fileName, fileName + customStr + '_V')).replace('.jpg', '.tiff').replace('.png', '.tiff')
    fML = (f.replace(fileName, fileName + customStr + '_ML')).replace('.jpg', '.tiff').replace('.png', '.tiff')
    fHH = (f.replace(fileName, fileName + customStr + '_HH')).replace('.jpg', '.tiff').replace('.png', '.tiff')
    fHS = (f.replace(fileName, fileName + customStr + '_HS')).replace('.jpg', '.tiff').replace('.png', '.tiff')

    # cA = Image.fromarray(cA)
    cH = Image.fromarray(cH)
    cV = Image.fromarray(cV)
    cD = Image.fromarray(cD)

    cA.save(join(path, fV))
    cH.save(join(path, fML))
    cV.save(join(path, fHH))
    cD.save(join(path, fHS))


# 统一大小
def augmentAndTrasformImage(f, mainFolder, trainFolder):
    try:
        img = Image.open(join(mainFolder, f))
    except:
        print('Error: Couldnt read the file {}. Make sure only images are present in the folder'.format(f))
        return None
    w, h = img.size
    if h > w:
        img = img.resize((rows, cols))
    else:
        img = img.resize((cols, rows))

    imgGray = img.convert('RGB')
    wdChk, htChk = imgGray.size
    if htChk > wdChk:
        imgGray = imgGray.rotate(-90, expand=1)
        print('training image rotated')
    transformImageAndSave(imgGray, f, '', trainFolder)


    return True


def createTrainingData(positiveImagePath, negativeImagePath, positiveTrainImagePath, negativeTrainImagePath):
    # 列表生成式，列举一个文件夹下是文件的所有文件名， if 用来滤除其他非当前文件夹文件。
    positiveImageFiles = [f for f in listdir(positiveImagePath) if (isfile(join(positiveImagePath, f)))]
    negativeImageFiles = [f for f in listdir(negativeImagePath) if (isfile(join(negativeImagePath, f)))]

    positiveCount = len(positiveImageFiles)
    negativeCount = len(negativeImageFiles)

    print('positive samples: ' + str(positiveCount))
    print('negative samples: ' + str(negativeCount))

    shutil.rmtree(positiveTrainImagePath)
    shutil.rmtree(negativeTrainImagePath)

    if not os.path.exists(positiveTrainImagePath):
        os.makedirs(positiveTrainImagePath)
    if not os.path.exists(negativeTrainImagePath):
        os.makedirs(negativeTrainImagePath)


    Knegative = 0
    Kpositive = 0

    # 提取摩尔纹图片特征
    for f in positiveImageFiles:
        ret = augmentAndTrasformImage(f, positiveImagePath, positiveTrainImagePath)
        if ret == None:
            continue
        Kpositive += 1

    # 提取正常图片特征
    for f in negativeImageFiles:
        ret = augmentAndTrasformImage(f, negativeImagePath, negativeTrainImagePath)
        if ret == None:
            continue
        Knegative += 1

    print('Total positive files after augmentation: ', Kpositive)
    print('Total negative files after augmentation: ', Knegative)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('positiveImages', type=str, help='Directory with positive (Moiré pattern) images.')
    parser.add_argument('negativeImages', type=str, help='Directory with negative (Normal) images.')
    parser.add_argument('train', type=int, help='0 = train, 1 = test')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(['./data/摩尔纹视频图片',
                          './data/10001training',
                          0]))

